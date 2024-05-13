# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .two_stage import TwoStageDetector
from .faster_rcnn import FasterRCNN

import os
import torch
import warnings
import mmcv
from collections import OrderedDict
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint, load_state_dict
from mmcv.parallel import MMDistributedDataParallel
from mmdet.core import distance2bbox
import torch.nn.functional as F

@DETECTORS.register_module()
class FasterRCNNLwF(FasterRCNN):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 ori_config_file=None,
                 ori_checkpoint_file=None,
                 kd_scale = 1e-1,
                 init_cfg=None):
        super(FasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        self.ori_checkpoint_file = ori_checkpoint_file
        self.ori_config_file = ori_config_file
        self.init_detector(ori_config_file, ori_checkpoint_file)
        self.kd_scale = kd_scale

    def init_detector(self, config, checkpoint_file):
        """Initialize detector from config file.

        Args:
            config (str): Config file path or the config
                object.
            checkpoint_file (str): Checkpoint path. If left as None, the model
                will not load any weights.

        Returns:
            nn.Module: The constructed detector.
        """
        assert os.path.isfile(checkpoint_file), '{} is not a valid file'.format(checkpoint_file)
        ##### init original model & frozen it #####
        # build model
        cfg = mmcv.Config.fromfile(config)
        cfg.model.pretrained = None
        ori_model = build_detector(
            cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))#test_cfg=cfg.test_cfg
        # load checkpoint
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        state_dict = checkpoint['state_dict']
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k,
                            v in checkpoint['state_dict'].items()}
        

        if hasattr(ori_model, 'module'):
            load_state_dict(ori_model.module, state_dict, strict = True, logger = None)
        else:
            load_state_dict(ori_model, state_dict, strict = True, logger = None)
        # load_checkpoint(ori_model, checkpoint_file)
        # set to eval mode
        ori_model.eval()
        ori_model.forward = ori_model.forward_dummy
        # set requires_grad of all parameters to False
        for param in ori_model.parameters():
            param.requires_grad = False

        self.ori_model = ori_model.to("cpu")
        print("original model loaded!!!!")

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()

        self.ori_model.to(img.device)
        x = self.extract_feat(img)
        with torch.no_grad():
            x_ori = self.ori_model.extract_feat(img)
        
        loss_kd = 0
        for i in range(len(x)):
            loss_kd = loss_kd + torch.mean((x[i] - x_ori[i]) ** 2) ** 0.5
        loss_kd = loss_kd * self.kd_scale
        losses.update({"loss_kd": loss_kd})

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses
    
