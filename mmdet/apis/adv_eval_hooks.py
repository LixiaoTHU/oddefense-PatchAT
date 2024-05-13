# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import collate, scatter
from mmcv.runner import Hook
from pycocotools.cocoeval import COCOeval
from torch.utils.data import Dataset

from mmdet import datasets
from mmdet.core import DistEvalHook
from torch.nn.modules.batchnorm import _BatchNorm

class AdvDistEvalHook(DistEvalHook):
    def __init__(self, *args, test_adv_cfg=None, **kwargs):
        super(AdvDistEvalHook, self).__init__(*args, **kwargs)
        self.test_adv_cfg=test_adv_cfg
        
    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        from mmdet.apis import multi_gpu_test

        # Changed results to self.results so that MMDetWandbHook can access
        # the evaluation results and log them to wandb.
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect,
            test_adv_cfg=self.test_adv_cfg)
        self.latest_results = results
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)

            # the key_score may be `None` so it needs to skip
            # the action to save the best checkpoint
            if self.save_best and key_score:
                self._save_ckpt(runner, key_score)

