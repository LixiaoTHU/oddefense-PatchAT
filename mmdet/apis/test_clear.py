# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import pickle
import shutil
import tempfile
import time, copy

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results
from mmcv.parallel import scatter
import cv2, os


def cal_adv(model, img, adv_sample, img_transform, test_adv_cfg):
    step_size = test_adv_cfg.get("step_size", 8)
    epsilon = test_adv_cfg.get("epsilon", 8)
    num_steps = test_adv_cfg.get("num_steps", 20)

    adv_type = test_adv_cfg.get("adv_type", "cls") 
    assert adv_type in ["cls", "reg", "cwa", "dag", "ours"]

    
    img_adv = img.detach().clone()
    img_adv.requires_grad_()

    if adv_type == "cwa":
        for i in range(len(adv_sample["img_metas"])):
            adv_sample["img_metas"][i]['cwa'] = True
    
    for i in range(len(adv_sample["img_metas"])):
        adv_sample["img_metas"][i]['adv_flag'] = True
    
    for step in range(num_steps):
        tmp = img_transform[0](img_adv)
        adv_sample['img'] = tmp

        # print(tmp.shape)
        # exit(0)
        loss_dict = model(**adv_sample, return_loss=True)
        # loss, _ = model.module._parse_losses(loss_dict)

        if adv_type == "cls":
            if isinstance(loss_dict['loss_cls'], list):
                loss = torch.cat(loss_dict['loss_cls'])
            else:
                loss = loss_dict['loss_cls']
            loss = torch.sum(loss)
            # loss_value = loss_dict['loss_rpn_cls']
            # loss_rpn_cls = sum(_loss.mean() for _loss in loss_value)
            # loss += loss_rpn_cls
        elif adv_type == "reg":
            if isinstance(loss_dict['loss_cls'], list):
                loss = torch.stack(loss_dict['loss_bbox'])
            else:
                loss = loss_dict['loss_bbox']
            loss = torch.sum(loss)
        elif adv_type == "cwa":
            if isinstance(loss_dict['loss_cls'], list):
                cls_loss = torch.sum(torch.cat(loss_dict['loss_cls']))
                reg_loss = torch.sum(torch.stack(loss_dict['loss_bbox']))
            else:
                cls_loss = loss_dict['loss_cls']
                reg_loss = loss_dict['loss_bbox']
                # loss_value = loss_dict['loss_rpn_cls']
                # loss_rpn_cls = sum(_loss.mean() for _loss in loss_value)
                # loss_value = loss_dict['loss_rpn_bbox']
                # loss_rpn_bbox = sum(_loss.mean() for _loss in loss_value)
            loss = cls_loss + reg_loss
        else:
            print("Not implement")
            exit(0)
        
        x_grad = torch.autograd.grad(loss, [img_adv], retain_graph=False)[0]
        eta = torch.sign(x_grad) * step_size

        img_adv = torch.min(torch.max(img_adv + eta, img - epsilon), img + epsilon)
        img_adv = torch.clamp(img_adv, 0.0, 255.0)
    
    for i in range(len(adv_sample["img_metas"])):
        adv_sample["img_metas"][i]['adv_flag'] = False

    return img_adv



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
    else:
        adv_flag = False

    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
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

            # save = True
            # if save:
            #     root = "/home/lixiao/ssd/workdir/oddefense/dataset/" + "dndetr_resnet_all"
            #     if not os.path.exists(root):
            #         os.mkdir(root)
            #     to_save = img_transform[1](img_adv)
            #     name = sample['img_metas'][0]["ori_filename"]
            #     cv2.imwrite(os.path.join(root, name), to_save.squeeze(0).permute(1,2,0).detach().cpu().numpy()[:,:,(2,1,0)])

            # root = "/home/lixiao/ssd/workdir/oddefense/dataset/" + "pascal_new/"
            # if not os.path.exists(root):
            #     os.mkdir(root)
            # to_save = img_transform[1](img_adv)
            # name = sample['img_metas'][0]["ori_filename"]
            # cv2.imwrite(os.path.join(root, name.split(".")[0] + ".png"), to_save.squeeze(0).permute(1,2,0).detach().cpu().numpy()[:,:,(2,1,0)])

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
            
            # alter = True
            # if alter:
            #     folders = ["none", "frcnn_resnet_all", "fcos_resnet_all", "dndetr_resnet_all", "frcnn_conv_all", "fcos_conv_all", "dndetr_conv_all"]
            #     root = "/home/lixiao/data3/workdir/oddefense/dataset/" + folders[6]
            #     name = data['img_metas']._data[0][0]["ori_filename"]
            #     newimg = cv2.imread(os.path.join(root, name), cv2.COLOR_BGR2RGB)
            #     newimg = torch.from_numpy(newimg).permute(2, 0, 1).unsqueeze(0).float()
            #     # newimg = img_transform[1](newimg)
            #     newimg[0, 0, :, :] = (newimg[0, 0, :, :] - 123.675) / 58.395
            #     newimg[0, 1, :, :] = (newimg[0, 1, :, :] - 116.28) / 57.12
            #     newimg[0, 2, :, :] = (newimg[0, 2, :, :] - 103.53) / 57.375
            #     data['img']._data[0] = newimg.to(data['img']._data[0].device)
                


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
