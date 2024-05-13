# Copyright (c) OpenMMLab. All rights reserved.
import math
import os.path as osp
import pickle
import random
import shutil
import sys
import tempfile
import time, copy

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results
from mmcv.parallel import scatter
import cv2, os
from torch import Tensor, nn, optim
from .patch import init_patch, load_patch
from torchvision.utils import save_image
from torch.nn import Module
from .train_universal_patch import train_universal_patch
from .patch_transformer import PatchTransformer
from .patch_applier import PatchApplier
from .get_bbox_patch import cal_adv
  



def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3,
                    test_adv_cfg=None):

    
    if test_adv_cfg is not None:
        adv_flag = test_adv_cfg.get("adv_flag", False)
    else:
        adv_flag = False
    
    model.eval()
    results = []
    dataset = data_loader.dataset
    PALETTE = getattr(dataset, 'PALETTE', None)
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        if adv_flag:
            sample = scatter(data, [torch.cuda.current_device()])[0]
            img = sample['img']

            img_mean = torch.from_numpy(sample['img_metas'][0]['img_norm_cfg']['mean']).to(img.device)
            img_mean = img_mean.unsqueeze(0).unsqueeze(2).unsqueeze(2)
            img_std = torch.from_numpy(sample['img_metas'][0]['img_norm_cfg']['std']).to(img.device)
            img_std = img_std.unsqueeze(0).unsqueeze(2).unsqueeze(2)
            

            img_transform = (lambda x: (x - img_mean) / img_std, lambda x: x * img_std + img_mean)
            
            img = img_transform[1](img)
            # adv_sample = copy.deepcopy(sample)
            adv_sample = sample
            img_adv = cal_adv(model, img, adv_sample, img_transform, test_adv_cfg)
            img_adv = img_transform[0](img_adv)

            # print(torch.max(torch.abs(img_adv - sample['img'])))

            sample.pop('gt_bboxes')
            sample.pop('gt_labels')
            if 'gt_masks' in sample:
                sample.pop('gt_masks')
            if 'gt_semantic_seg' in sample:
                sample.pop('gt_semantic_seg')
            sample['img_metas'] = [sample['img_metas']]
            sample['img'] = [img_adv.detach()]
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **sample)
        else:
            data.pop('gt_bboxes')
            data.pop('gt_labels')
            if 'gt_masks' in data:
                data.pop('gt_masks')
            if 'gt_semantic_seg' in data:
                data.pop('gt_semantic_seg')
            data['img_metas'] = [data['img_metas']]
            data['img'] = [data['img']]
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)

        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    bbox_color=PALETTE,
                    text_color=PALETTE,
                    mask_color=PALETTE,
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        # This logic is only used in panoptic segmentation test.
        elif isinstance(result[0], dict) and 'ins_results' in result[0]:
            for j in range(len(result)):
                bbox_results, mask_results = result[j]['ins_results']
                result[j]['ins_results'] = (bbox_results,
                                            encode_mask_results(mask_results))

        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False, test_adv_cfg=None):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """

    if test_adv_cfg is not None:
        adv_flag = test_adv_cfg.get("adv_flag", False)
        universal_patch_config = test_adv_cfg.get("universal_patch_config", None)
        bbox_patch_config = test_adv_cfg.get("bbox_patch_config", None)
        assert not (universal_patch_config['patch_flag'] and bbox_patch_config['patch_flag'])

    else:
        adv_flag = False

    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    
    if adv_flag and universal_patch_config['patch_flag']:
        if universal_patch_config['eval_patch']:
            patch = load_patch(universal_patch_config['patch_path'])
            patch_transformer = PatchTransformer(scale=universal_patch_config['scale_factor'], patch_type=universal_patch_config['patch_type'])
            patch_applier = PatchApplier()
        else:
            patch, patch_transformer, patch_applier = train_universal_patch(model, data_loader, universal_patch_config, rank, world_size)

    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        if adv_flag:
            sample = scatter(data, [torch.cuda.current_device()])[0]
            img = sample['img']
            img_mean = torch.from_numpy(sample['img_metas'][0]['img_norm_cfg']['mean']).to(img.device)
            img_mean = img_mean.unsqueeze(0).unsqueeze(2).unsqueeze(2)
            img_std = torch.from_numpy(sample['img_metas'][0]['img_norm_cfg']['std']).to(img.device)
            img_std = img_std.unsqueeze(0).unsqueeze(2).unsqueeze(2)
            

            img_transform = (lambda x: (x - img_mean) / img_std, lambda x: x * img_std + img_mean)
            
            adv_sample = sample

            if adv_flag:
                if universal_patch_config['patch_flag']:
                    target = sample['gt_bboxes']
                    lab = sample['gt_labels']
                    target_len = [len(t) for t in target]
                    target = torch.cat(tuple(target), dim=0)
                    lab = torch.cat(tuple(lab), dim=0)
                    if target.shape[0] == 0:
                        img_adv = img
                    else:
                        adv_batch, msk_batch = patch_transformer(patch, target, (img.shape[-2], img.shape[-1]), lab)
                        img_adv = patch_applier(img, target_len, adv_batch, msk_batch)
                else:
                    
                    img = img_transform[1](img)
                    img_adv = cal_adv(model, img, adv_sample, img_transform, bbox_patch_config)
                    img_adv = img_transform[0](img_adv)

            sample.pop('gt_bboxes')
            sample.pop('gt_labels')
            if 'gt_masks' in sample:
                sample.pop('gt_masks')
            if 'gt_semantic_seg' in sample:
                sample.pop('gt_semantic_seg')

            sample['img_metas'] = [sample['img_metas']]
            sample['img'] = [img_adv.detach()]
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **sample)
        else:
            data.pop('gt_bboxes')
            data.pop('gt_labels')
            if 'gt_masks' in data:
                data.pop('gt_masks')
            if 'gt_semantic_seg' in data:
                data.pop('gt_semantic_seg')
                


            data['img_metas'] = [data['img_metas']]
            data['img'] = [data['img']]
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)
        

            # encode mask results
        with torch.no_grad():
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                            for bbox_results, mask_results in result]
            # This logic is only used in panoptic segmentation test.
            elif isinstance(result[0], dict) and 'ins_results' in result[0]:
                for j in range(len(result)):
                    bbox_results, mask_results = result[j]['ins_results']
                    result[j]['ins_results'] = (
                        bbox_results, encode_mask_results(mask_results))

        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
