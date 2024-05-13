# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .two_stage import TwoStageDetector
from .faster_rcnn import FasterRCNN
import torch

@DETECTORS.register_module()
class FasterRCNNL2(FasterRCNN):

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None,
                 L2_regularization=False,
                 reg_coef=1e2):
        super(FasterRCNNL2, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        
        self.L2_regularization = L2_regularization
        self.params = {n: p for n, p in self.named_parameters() if p.requires_grad}  # For convenience
        self.reg_coef = reg_coef
        self.task_param = dict()

    def calculate_importance(self, dataloader):
        # Use an identity importance so it is an L2 regularization.
        importance = {}
        for n, p in self.params.items():
            importance[n] = p.clone().detach().fill_(1)  # Identity
        return importance
    
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        losses = super(FasterRCNNL2, self).forward_train(img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore,
                      gt_masks,
                      proposals,
                      **kwargs)

        if self.L2_regularization:

            # Calculate the importance of weights for current task
            importance = self.calculate_importance(None)
            self.regularization_terms = {'importance':importance, 'task_param': self.task_param}
            
            importance = self.regularization_terms['importance']
            task_param = self.regularization_terms['task_param']
            if len(task_param) > 0:
                task_reg_loss = 0
                for n, p in self.named_parameters():
                    if p.requires_grad:
                        task_reg_loss += (importance[n] * (p - task_param[n]) ** 2).sum()
                        # print("gg: ", torch.sum(task_param[n]))
                        # break
                l2_reg_loss = task_reg_loss
                l2_reg_loss = self.reg_coef * l2_reg_loss
                # print(l2_reg_loss)
                losses.update({"l2_reg_loss": l2_reg_loss})
            else:
                # Backup the weight of current task
                for n, p in self.named_parameters():
                    self.task_param[n] = p.clone().detach()


        return losses

            
        
