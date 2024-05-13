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
class FasterRCNNIncre(FasterRCNN):
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
                 init_cfg=None):
        super(FasterRCNNIncre, self).__init__(
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
        checkpoint = torch.load(checkpoint_file, map_location=None)
        state_dict = checkpoint['state_dict']
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k,
                          v in checkpoint['state_dict'].items()}
        
        # for name in state_dict:
        #     print(name)
        
        # for name in ori_model.state_dict():
        #     print(name)
        # exit(0)
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

        self.ori_model = ori_model
        print("original model loaded!!!!")

    
    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        # x = self.extract_feat(img)
        x = self.ori_model.extract_feat(img)

        if proposals is None:  # Faster-RCNN: None
            # proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
            proposal_list = self.ori_model.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals
        print("")
        out = self.ori_model.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)
        return self.ori_model.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)
