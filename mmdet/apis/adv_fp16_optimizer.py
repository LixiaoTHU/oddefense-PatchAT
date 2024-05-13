# Copyright (c) OpenMMLab. All rights reserved.
import copy
import logging
from collections import defaultdict
from itertools import chain
from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils import clip_grad

from mmcv.utils import TORCH_VERSION, _BatchNorm, digit_version
from mmcv.runner import HOOKS, Hook, Fp16OptimizerHook

try:
    # If PyTorch version >= 1.6.0, torch.cuda.amp.GradScaler would be imported
    # and used; otherwise, auto fp16 will adopt mmcv's implementation.
    from torch.cuda.amp import GradScaler
except ImportError:
    pass


@HOOKS.register_module()
class Fp16AdvOptimizerHook(Fp16OptimizerHook):
    """A hook contains custom operations for the optimizer.

    Args:
        grad_clip (dict, optional): A config dict to control the clip_grad.
            Default: None.
        detect_anomalous_params (bool): This option is only used for
            debugging which will slow down the training speed.
            Detect anomalous parameters that are not included in
            the computational graph with `loss` as the root.
            There are two cases

                - Parameters were not used during
                  forward pass.
                - Parameters were not used to produce
                  loss.
            Default: False.
    """
    def __init__(self,
                grad_clip: Optional[dict] = None,
                coalesce: bool = True,
                bucket_size_mb: int = -1,
                loss_scale: Union[float, str, dict] = 512.,
                distributed: bool = True):
        self.grad_clip = grad_clip
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb
        self.distributed = distributed
        self._scale_update_param = None
        if loss_scale == 'dynamic':
            self.loss_scaler = GradScaler()
        elif isinstance(loss_scale, float):
            self._scale_update_param = loss_scale
            self.loss_scaler = GradScaler(init_scale=loss_scale)
        elif isinstance(loss_scale, dict):
            self.loss_scaler = GradScaler(**loss_scale)
        else:
            raise ValueError('loss_scale must be of type float, dict, or '
                                f'"dynamic", got {loss_scale}')



    def after_train_iter(self, runner) -> None:
        """Backward optimization steps for Mixed Precision Training. For
        dynamic loss scaling, please refer to
        https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler.

        1. Scale the loss by a scale factor.
        2. Backward the loss to obtain the gradients.
        3. Unscale the optimizer’s gradient tensors.
        4. Call optimizer.step() and update scale factor.
        5. Save loss_scaler state_dict for resume purpose.
        """
        # clear grads of last iteration
        runner.model.zero_grad()
        runner.optimizer.zero_grad()

        loss = self.loss_scaler.scale(runner.outputs['loss'])


        model = runner.model.module
        epsilon = model.epsilon
        noise_transform = (lambda x: x / model.img_std, lambda x: x * model.img_std)
        
        if model.adv_type == "mtd":
            loss.backward(retain_graph=True)
            losses_cls, losses_bbox = model.mtd_lossdict["loss_cls"], model.mtd_lossdict["loss_bbox"]
            if model.mtd_lossdict["sum"]:
                losses_generate = torch.sum(model.mtd_index * losses_cls + (~model.mtd_index) * losses_bbox)
            else:
                losses_generate = torch.mean(model.mtd_index * losses_cls + (~model.mtd_index) * losses_bbox)
            grad_mtd = torch.autograd.grad(losses_generate, [model.aux_img])[0].detach()

            adv_noise = noise_transform[1](model.adv_noise)
            adv_noise = torch.clamp(adv_noise + epsilon * torch.sign(grad_mtd), -epsilon, epsilon).detach_()
            adv_noise = noise_transform[0](adv_noise)
        else:
            loss.backward()

            # calculate noise
            grad_adv = model.aux_img.grad.detach()

            adv_noise = noise_transform[1](model.adv_noise)
            adv_noise = torch.clamp(adv_noise + epsilon * torch.sign(grad_adv), -epsilon, epsilon).detach_()
            adv_noise = noise_transform[0](adv_noise)
            model.aux_img.grad.data.zero_()


        model.adv_noise = adv_noise

        self.loss_scaler.unscale_(runner.optimizer)
        # grad clip
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                            runner.outputs['num_samples'])
        # backward and update scaler
        self.loss_scaler.step(runner.optimizer)
        self.loss_scaler.update(self._scale_update_param)

        # save state_dict of loss_scaler
        runner.meta.setdefault(
            'fp16', {})['loss_scaler'] = self.loss_scaler.state_dict()



