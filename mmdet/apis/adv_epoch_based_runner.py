# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import platform
import shutil
import time, copy
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

import mmcv
from mmcv.runner import (BaseRunner, RUNNERS, EpochBasedRunner,
                        save_checkpoint, get_host_info)


@RUNNERS.register_module()
class AdvEpochBasedRunner(EpochBasedRunner):
    """Adv Epoch-based Runner.

    This runner train models epoch by epoch.

    Adapted from mmcv.runner
    """
    def __init__(self, freeze_dict = None, **kwargs):
        super(AdvEpochBasedRunner, self).__init__(**kwargs)
        self.freeze_dict = freeze_dict


    def run_iter(self, data_batch: Any, train_mode: bool, **kwargs) -> None:
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(data_batch, self.optimizer,
                                            **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader

        if self.freeze_dict is not None and self._epoch in self.freeze_dict:
                self.model.module.backbone.frozen_stages = self.freeze_dict[self._epoch]

        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition


        self.free_m = self.model.module.free_m
        for i, data_batch in enumerate(self.data_loader):
            # self._inner_iter = i
            self.data_batch = data_batch # Not changed by the model

            # gg = data_batch['img']._data[0] + torch.Tensor([123.675, 116.28, 103.53]).unsqueeze(0).unsqueeze(2).unsqueeze(2)
            # print(torch.min(gg))
            # print(torch.max(gg))
            # print(torch.sum(gg > 255) / torch.sum(gg > -10000))
            # print(torch.sum(gg < 0) / torch.sum(gg > -10000))

            for j in range(self.free_m):
                self._inner_iter = i * self.free_m + j
                self.call_hook('before_train_iter')
                self.run_iter(data_batch, train_mode=True, **kwargs)
                self.call_hook('after_train_iter')
                
            del self.data_batch
            self._iter += 1
            # if self._iter > 2:
            #     break

        self.call_hook('after_train_epoch')
        self._epoch += 1

    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self.data_batch = data_batch
            self._inner_iter = i
            self.call_hook('before_val_iter')
            self.run_iter(data_batch, train_mode=False)
            self.call_hook('after_val_iter')
            del self.data_batch
        self.call_hook('after_val_epoch')

    def run(self,
            data_loaders: List[DataLoader],
            workflow: List[Tuple[str, int]],
            max_epochs: Optional[int] = None,
            **kwargs) -> None:
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

