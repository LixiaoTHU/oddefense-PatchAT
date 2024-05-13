# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import random

import mmcv
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import cv2
from mmcv.runner import BaseModule, auto_fp16
from PIL import Image

from mmdet.core.visualization import imshow_det_bboxes


class BaseDetector(BaseModule, metaclass=ABCMeta):
    """Base class for detectors."""

    def __init__(self, init_cfg=None, adv_cfg=None):
        super(BaseDetector, self).__init__(init_cfg)
        self.fp16_enabled = False
        self.adv_cfg = adv_cfg
        if self.adv_cfg is not None:
            self.adv_flag = self.adv_cfg.get("adv_flag", False)
            self.free_m = self.adv_cfg.get("free_m", False)
            self.adv_type = self.adv_cfg.get("adv_type", "all")
            assert self.adv_type in ["all", "mtd", "cwa", "com"]
            self.linf_config = self.adv_cfg.get("linf_config", None)
            self.patch_config = self.adv_cfg.get("patch_config", None)

            # linf config
            self.linf_epsilon = self.linf_config.get("epsilon", 4)
            self.linf_flag = self.linf_config.get("linf_flag", False)

            # patch config
            self.patch_flag = self.patch_config.get("patch_flag", False)
            self.patch_epsilon = self.patch_config.get("epsilon", 8)
            self.patch_step_size = self.patch_config.get("patch_step_size", 8)
            self.patch_scale_factor = self.patch_config.get("scale_factor", 1/10/1.414*2)
            self.without_gradient_info = self.patch_config.get("without_gradient_info", False)


            self.adv_noise = None
            self.adv_patch = None
            self.aux_img = None

            self.img_norm = None
            self.img_std = None
            self.img_transform = None

            self.cnt = 0
        else:
            self.adv_flag = False


    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self, 'neck') and self.neck is not None

    # TODO: these properties need to be carefully handled
    # for both single stage & two stage detectors
    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self, 'roi_head') and self.roi_head.with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self, 'roi_head') and self.roi_head.with_bbox)
                or (hasattr(self, 'bbox_head') and self.bbox_head is not None))

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return ((hasattr(self, 'roi_head') and self.roi_head.with_mask)
                or (hasattr(self, 'mask_head') and self.mask_head is not None))

    @abstractmethod
    def extract_feat(self, imgs):
        """Extract features from images."""
        pass

    def extract_feats(self, imgs):
        """Extract features from multiple images.

        Args:
            imgs (list[torch.Tensor]): A list of images. The images are
                augmented from the same image but in different ways.

        Returns:
            list[torch.Tensor]: Features of different images
        """
        assert isinstance(imgs, list)
        return [self.extract_feat(img) for img in imgs]

    def forward_train(self, imgs, img_metas, **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys, see
                :class:`mmdet.datasets.pipelines.Collect`.
            kwargs (keyword arguments): Specific to concrete implementation.
        """
        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        
        batch_input_shape = tuple(imgs[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape

    async def async_simple_test(self, img, img_metas, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def simple_test(self, img, img_metas, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        pass

    async def aforward_test(self, *, img, img_metas, **kwargs):
        for var, name in [(img, 'img'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(img)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(img)}) '
                             f'!= num of image metas ({len(img_metas)})')
        # TODO: remove the restriction of samples_per_gpu == 1 when prepared
        samples_per_gpu = img[0].size(0)
        assert samples_per_gpu == 1

        if num_augs == 1:
            return await self.async_simple_test(img[0], img_metas[0], **kwargs)
        else:
            raise NotImplementedError

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_metas, **kwargs)


    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, with_patch = False, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            return self.onnx_export(img[0], img_metas[0])
        if return_loss and not with_patch:
            return self.forward_train(img, img_metas, **kwargs)
        elif return_loss and with_patch:
            return self.forward_patch(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        # If the loss_vars has different length, GPUs will wait infinitely
        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            dist.all_reduce(log_var_length)
            message = (f'rank {dist.get_rank()}' +
                       f' len(log_vars): {len(log_vars)}' + ' keys: ' +
                       ','.join(log_vars.keys()))
            assert log_var_length == len(log_vars) * dist.get_world_size(), \
                'loss log variables are different across GPUs!\n' + message

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars
    
    def _parse_loss_mtd(self, losses, log_vars):
        self.mtd_lossdict = dict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, list):
                self.mtd_lossdict['sum'] = True
                self.mtd_lossdict[loss_name] = torch.stack(loss_value).squeeze()
            elif isinstance(loss_value, torch.Tensor):
                self.mtd_lossdict['sum'] = False # mean
                self.mtd_lossdict[loss_name] = loss_value
            else:
                print("Not implement")
                exit(0)


    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        self.data = data

        assert(not (self.adv_flag==True and self.patch_flag==False and self.linf_flag==False))

        if self.adv_flag:

            # init adv_noise
            if self.adv_noise is None and self.adv_patch is None:

                device = data['img'].device
                self.adv_noise = torch.zeros(data['img'].shape, device=device) if self.linf_flag else None
                self.adv_patch = torch.zeros(data['img'].shape, device=device) if self.patch_flag else None
                self.aux_img = torch.zeros(data['img'].shape, device=device)
                self.img_norm = torch.Tensor(data['img_metas'][0]['img_norm_cfg']['mean']).to(device)
                self.img_std = torch.Tensor(data['img_metas'][0]['img_norm_cfg']['std']).to(device)
                self.img_norm = self.img_norm.unsqueeze(0).unsqueeze(2).unsqueeze(2)
                self.img_std = self.img_std.unsqueeze(0).unsqueeze(2).unsqueeze(2)
                
                self.img_transform=(lambda x: (x - self.img_norm) / self.img_std, lambda x: x * self.img_std + self.img_norm)
            
            img_ori = data['img']
            device = data['img'].device
            mask_ = torch.zeros_like(img_ori)
            self.aux_img = img_ori

            if self.patch_flag:
                boxes_batch = data['gt_bboxes']
                
                self.adv_patch = torch.nn.functional.interpolate(self.adv_patch, size=(img_ori.shape[-2], img_ori.shape[-1]))

                img_ori.requires_grad_(True)
                losses = self(**data)
                loss = losses["loss_cls"] + losses["loss_bbox"]
                grad_mtd = torch.autograd.grad(loss, [img_ori])[0].detach()

                if grad_mtd is not None:
                    grad_mtd=torch.norm(grad_mtd, dim=1)
                    for idx, box in enumerate(boxes_batch):
                        grad = grad_mtd[idx]
                        for b in box:
                            choices = random.choices([0, 1], weights=[1,1], k=1)
                            if choices[0] == 0:
                                continue
                            y1, x1, y2, x2 = b
                            w = x2 - x1
                            h = y2 - y1
                            s = int(torch.sqrt(w**2+h**2)*self.patch_scale_factor/2)
                            offset_x = torch.normal(mean=0, std=w * self.patch_scale_factor)
                            offset_y = torch.normal(mean=0, std=h * self.patch_scale_factor)
                            x_ = int(max(min((x1+x2)/2 + offset_x, img_ori.shape[-2]-s), s))
                            y_ = int(max(min((y1+y2)/2 + offset_y, img_ori.shape[-1]-s), s))
                            if s >= 4:
                                s = s//4 * 4
                                patch_ = grad[x_-s:x_+s, y_-s:y_+s].clone()
                                patch_ = patch_.unsqueeze(0)
                                pool = torch.nn.AvgPool2d(s//4,stride=s//4,padding =0)
                                patch = pool(patch_)
                                patch = patch.squeeze(0)

                                if self.without_gradient_info:
                                    indices = torch.randperm(64)[:32]
                                else:
                                    values, indices = torch.topk(patch.view(-1), k=32,sorted = False)
                                patch_mask_ = torch.zeros_like(patch)
                                patch_mask_.view(-1)[indices] = 1
                                target_size = (s//4,s//4)
                                patch_mask = torch.kron(patch_mask_, torch.ones(target_size).to(device))
                                s = patch_mask.shape[-1]//2
                                mask_[idx, :, x_-s:x_+s, y_-s:y_+s] = patch_mask
                            else :
                                mask_[idx, :, x_-s:x_+s, y_-s:y_+s] = 1
                self.aux_img = self.aux_img + self.adv_patch*mask_
            if self.linf_flag:
                self.adv_noise = torch.nn.functional.interpolate(self.adv_noise, size=(img_ori.shape[-2], img_ori.shape[-1]))
                self.aux_img = self.aux_img + self.adv_noise*(1-mask_)

            '''gg = self.img_transform[1](self.aux_img)

            self.cnt += 1
            if self.cnt < 400:
                if self.cnt%10==0:
                    save_img = gg[1].permute(1,2,0).squeeze(0).detach().cpu().numpy()[:,:,(2,1,0)].copy()
                    box = data['gt_bboxes'][1]
                    for b in box:
                        y1, x1, y2, x2 = b
                        cv2.line(save_img,(int(y1),int(x1)),(int(y1),int(x2)),(0,0,255),2)
                        cv2.line(save_img,(int(y1),int(x1)),(int(y2),int(x1)),(0,0,255),2)
                        cv2.line(save_img,(int(y2),int(x2)),(int(y1),int(x2)),(0,0,255),2)
                        cv2.line(save_img,(int(y2),int(x2)),(int(y2),int(x1)),(0,0,255),2)
                    cv2.imwrite("/home/zhuyiming/ssd/grad_patch/mmdetection/out/raw%s.png" % (int(self.cnt)), save_img)
            else:
                exit(0)'''

            self.aux_img = self.img_transform[0](torch.clamp(self.img_transform[1](self.aux_img), min=0, max=255).detach_())
            self.aux_img.requires_grad_()

            

            data['img'] = self.aux_img
            # print(torch.max(torch.abs(data['img'] - img_ori)))

            if self.adv_type == "cwa":
                for i in range(len(data["img_metas"])):
                    data["img_metas"][i]['cwa'] = True
            elif self.adv_type == "mtd":
                for i in range(len(data["img_metas"])):
                    data["img_metas"][i]['mtd'] = True

            losses = self(**data)
            loss, log_vars = self._parse_losses(losses)
            if self.adv_type == "mtd":
                self._parse_loss_mtd(losses, log_vars)
                self.mtd_index = self.mtd_lossdict['loss_cls'] > self.mtd_lossdict['loss_bbox']
            elif self.adv_type == "all":
                self.head_loss = losses["loss_cls"] + losses["loss_bbox"]

            outputs = dict(
                loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        else:
            losses = self(**data)
            loss, log_vars = self._parse_losses(losses)

            outputs = dict(
                loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))
        
        return outputs

    def val_step(self, data, optimizer=None):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color=(72, 101, 241),
                    text_color=(72, 101, 241),
                    mask_color=None,
                    thickness=2,
                    font_size=13,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None
            thickness (int): Thickness of lines. Default: 2
            font_size (int): Font size of texts. Default: 13
            win_name (str): The window name. Default: ''
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        img = imshow_det_bboxes(
            img,
            bboxes,
            labels,
            segms,
            class_names=self.CLASSES,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            mask_color=mask_color,
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            return img

    def onnx_export(self, img, img_metas):
        raise NotImplementedError(f'{self.__class__.__name__} does '
                                  f'not support ONNX EXPORT')
