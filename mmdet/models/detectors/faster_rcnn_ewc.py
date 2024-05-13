# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .two_stage import TwoStageDetector
from .faster_rcnn_l2 import FasterRCNNL2
import torch

@DETECTORS.register_module()
class FasterRCNNEWC(FasterRCNN):

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None,
                 EWC_regularization=False,
                 reg_coef=1e2):
        super(FasterRCNNEWC, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        
        self.EWC_regularization = EWC_regularization
        self.params = {n: p for n, p in self.named_parameters() if p.requires_grad}  # For convenience
        self.reg_coef = reg_coef
        self.task_param = dict()



    def calculate_importance(self, dataloader):
        # Update the diag fisher information
        # There are several ways to estimate the F matrix.
        # We keep the implementation as simple as possible while maintaining a similar performance to the literature.
        print('Computing EWC')

        # Initialize the importance matrix
        importance = {}
        for n, p in self.params.items():
            importance[n] = p.clone().detach().fill_(0)  # zero initialized


        mode = self.training
        self.eval()

        # Accumulate the square of gradients
        for i, (input, target, task) in enumerate(dataloader):
            if self.gpu:
                input = input.cuda()
                target = target.cuda()

            preds = self.forward(input)

            # Sample the labels for estimating the gradients
            # For multi-headed model, the batch of data will be from the same task,
            # so we just use task[0] as the task name to fetch corresponding predictions
            # For single-headed model, just use the max of predictions from preds['All']
            task_name = task[0] if self.multihead else 'All'

            # The flag self.valid_out_dim is for handling the case of incremental class learning.
            # if self.valid_out_dim is an integer, it means only the first 'self.valid_out_dim' dimensions are used
            # in calculating the loss.
            pred = preds[task_name] if not isinstance(self.valid_out_dim, int) else preds[task_name][:,:self.valid_out_dim]
            ind = pred.max(1)[1].flatten()  # Choose the one with max

            # - Alternative ind by multinomial sampling. Its performance is similar. -
            # prob = torch.nn.functional.softmax(preds['All'],dim=1)
            # ind = torch.multinomial(prob, 1).flatten()

            loss = self.criterion(preds, ind, task, regularization=False)
            self.model.zero_grad()
            loss.backward()
            for n, p in importance.items():
                if self.params[n].grad is not None:  # Some heads can have no grad if no loss applied on them.
                    p += ((self.params[n].grad ** 2) * len(input) / len(dataloader))

        self.train(mode=mode)

        return importance

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
        losses = super(FasterRCNNEWC, self).forward_train(img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore,
                      gt_masks,
                      proposals,
                      **kwargs)

        if self.EWC_regularization:

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

                EWC_reg_loss = task_reg_loss
                EWC_reg_loss = self.reg_coef * EWC_reg_loss
                losses.update({"EWC_reg_loss": EWC_reg_loss})
            else:
                # Backup the weight of current task
                for n, p in self.named_parameters():
                    self.task_param[n] = p.clone().detach()


        return losses

            
        
