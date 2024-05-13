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
from mmcv.runner import HOOKS, Hook, OptimizerHook

try:
    # If PyTorch version >= 1.6.0, torch.cuda.amp.GradScaler would be imported
    # and used; otherwise, auto fp16 will adopt mmcv's implementation.
    from torch.cuda.amp import GradScaler
except ImportError:
    pass


@HOOKS.register_module()
class AdvOptimizerHook(OptimizerHook):
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
                 detect_anomalous_params: bool = False
                 ):
        self.grad_clip = grad_clip
        self.detect_anomalous_params = detect_anomalous_params

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip)

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        if self.detect_anomalous_params:
            self.detect_anomalous_parameters(runner.outputs['loss'], runner)
        
        model = runner.model.module
        epsilon = model.linf_epsilon
        patch_step_size = model.patch_step_size
        patch_clip = model.patch_epsilon
        noise_transform = (lambda x: x / model.img_std, lambda x: x * model.img_std)


        if model.adv_type == "mtd":
            runner.outputs['loss'].backward(retain_graph=True)
            losses_cls, losses_bbox = model.mtd_lossdict["loss_cls"], model.mtd_lossdict["loss_bbox"]
            if model.mtd_lossdict["sum"]:
                losses_generate = torch.sum(model.mtd_index * losses_cls + (~model.mtd_index) * losses_bbox)
            else:
                losses_generate = torch.mean(model.mtd_index * losses_cls + (~model.mtd_index) * losses_bbox)
            grad_mtd = torch.autograd.grad(losses_generate, [model.aux_img])[0].detach()
            adv_noise = noise_transform[1](model.adv_noise)
            adv_noise = torch.clamp(adv_noise + epsilon * torch.sign(grad_mtd), -epsilon, epsilon).detach_()
            adv_noise = noise_transform[0](adv_noise)
        elif model.adv_type == "all":
            runner.outputs['loss'].backward(retain_graph=True)
            losses_generate = model.head_loss
            grad_mtd = torch.autograd.grad(losses_generate, [model.aux_img])[0].detach()
            if model.patch_flag:
                adv_patch = noise_transform[1](model.adv_patch)
                adv_patch = torch.clamp(adv_patch + patch_step_size * torch.sign(grad_mtd), -patch_clip, patch_clip).detach_()
                adv_patch = noise_transform[0](adv_patch)
            if model.linf_flag:
                adv_noise = noise_transform[1](model.adv_noise)
                adv_noise = torch.clamp(adv_noise + epsilon * torch.sign(grad_mtd), -epsilon, epsilon).detach_()
                adv_noise = noise_transform[0](adv_noise)
        elif model.adv_type == "com":
            runner.outputs['loss'].backward()

            # calculate noise
            grad_adv = model.aux_img.grad.detach()

            adv_noise = noise_transform[1](model.adv_noise)
            adv_noise = torch.clamp(adv_noise + epsilon * torch.sign(grad_adv), -epsilon, epsilon).detach_()
            adv_noise = noise_transform[0](adv_noise)
            model.aux_img.grad.data.zero_()
        else:
            runner.outputs['loss'].backward()

            # calculate noise
            grad_adv = model.aux_img.grad.detach()

            adv_noise = noise_transform[1](model.adv_noise)
            adv_noise = torch.clamp(adv_noise + epsilon * torch.sign(grad_adv), -epsilon, epsilon).detach_()
            adv_noise = noise_transform[0](adv_noise)
            model.aux_img.grad.data.zero_()

            # pgd = True
            # if pgd:
            #     grad_adv = torch.autograd.grad(runner.outputs['loss'], [model.aux_img])[0]

            #     # calculate noise
            #     # grad_adv = model.aux_img.grad.detach()

            #     adv_noise = noise_transform[1](model.adv_noise)
            #     adv_noise = torch.clamp(adv_noise + epsilon * torch.sign(grad_adv), -epsilon, epsilon).detach_()
            #     adv_noise = noise_transform[0](adv_noise)

            #     data = model.data

            #     num_steps = 1
            #     for step in range(num_steps):
            #         pgd_img = model.aux_img + adv_noise
            #         pgd_img = model.img_transform[0](torch.clamp(model.img_transform[1](pgd_img), min=0, max=255).detach_())
            #         pgd_img.requires_grad_()
            #         data['img'] = pgd_img

            #         # runner.optimizer.zero_grad()
            #         losses = model(**data)
            #         loss, log_vars = model._parse_losses(losses)
            #         loss.backward()
            #         adv_noise = noise_transform[1](adv_noise)
            #         adv_noise = torch.clamp(adv_noise + epsilon * torch.sign(pgd_img.grad), -epsilon, epsilon).detach_()
            #         adv_noise = noise_transform[0](adv_noise)
            #     runner.outputs = dict(
            #                     loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        if model.linf_flag:
            model.adv_noise = adv_noise
        if model.patch_flag:
            model.adv_patch = adv_patch

        # print("opt:", torch.sum(torch.abs(model.adv_noise)))

        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
        runner.optimizer.step()


