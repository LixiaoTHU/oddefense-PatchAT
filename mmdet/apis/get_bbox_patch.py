import math
from .patch_transformer import PatchTransformer
from .save_img import save_img
import os.path as osp
import sys
import torch

def cal_adv(model, img, adv_sample, img_transform, bbox_patch_config):
    adv_type = bbox_patch_config.get("adv_type", "cls")
    step_size = bbox_patch_config.get("step_size", 2)
    epsilon = bbox_patch_config.get("epsilon", 255)
    num_steps = bbox_patch_config.get("num_steps", 200)
    scale_factor = bbox_patch_config.get("scale_factor", 0.2)
    patch_type = bbox_patch_config.get("patch_type", "square")

    if adv_type == "cwa":
        for i in range(len(adv_sample["img_metas"])):
            adv_sample["img_metas"][i]['cwa'] = True
    
    for i in range(len(adv_sample["img_metas"])):
        adv_sample["img_metas"][i]['adv_flag'] = True


    device = adv_sample['img'].device
    img_adv = img.detach()
    img_adv.requires_grad_()

    boxes_batch = adv_sample['gt_bboxes']
    mask_patch = torch.zeros(adv_sample['img'].shape, device=device)
    if patch_type == "round":
        for idx, box in enumerate(boxes_batch):
            x, y = torch.meshgrid([torch.arange(0, adv_sample['img'].shape[-2]), torch.arange(0, adv_sample['img'].shape[-1])])
            for b in box:
                y1, x1, y2, x2 = b
                w = x2 - x1
                h = y2 - y1
                s = int(torch.sqrt(w**2+h**2)*scale_factor)
                offset_x = torch.normal(mean=0, std=w * scale_factor)
                offset_y = torch.normal(mean=0, std=h * scale_factor)
                x_ = int(max(min((x1+x2)/2 + offset_x, x2-s), x1+s))
                y_ = int(max(min((y1+y2)/2 + offset_y, y2-s), y1+s))
                pi = torch.tensor(math.pi)
                d = s/torch.sqrt(pi)
                mask = (torch.sqrt((x - x_)**2 + (y - y_)**2) <= d).to(device)
                mask_patch[idx, :] += mask
        mask_patch = torch.clamp(mask_patch, 0, 1)
    elif patch_type == "square":
        for idx, box in enumerate(boxes_batch):
            for b in box:
                y1, x1, y2, x2 = b
                w = x2 - x1
                h = y2 - y1
                s = int(torch.sqrt(w**2+h**2)*scale_factor/2)
                offset_x = torch.normal(mean=0, std=w * scale_factor)
                offset_y = torch.normal(mean=0, std=h * scale_factor)
                x_ = int(max(min((x1+x2)/2 + offset_x, x2-s), x1+s))
                y_ = int(max(min((y1+y2)/2 + offset_y, y2-s), y1+s))
                mask_patch[idx, :,x_-s:x_+s, y_-s:y_+s] = 1.0




    patch = torch.zeros(adv_sample['img'].shape, device=device)
    for step in range(num_steps):
        
        tmp = img_transform[0](img_adv)
        adv_sample['img'] = tmp

        loss_dict = model(**adv_sample, return_loss=True)

        if adv_type == "cls":
            if isinstance(loss_dict['loss_cls'], list):
                loss = torch.cat(loss_dict['loss_cls'])
            else:
                loss = loss_dict['loss_cls']
            loss = torch.sum(loss)
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
            loss = cls_loss + reg_loss
        else:
            print("Not implement")
            exit(0)
        
        x_grad = torch.autograd.grad(loss, [img_adv], retain_graph=False)[0].detach()
        patch = torch.clamp(patch + torch.sign(x_grad) * step_size, -epsilon, epsilon)
        img_adv = torch.clamp(img_adv + patch*mask_patch, 0.0, 255.0)
    

    for i in range(len(adv_sample["img_metas"])):
        adv_sample["img_metas"][i]['adv_flag'] = False
    
    return img_adv

