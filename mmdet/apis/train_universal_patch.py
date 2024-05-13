import os.path as osp
import sys
import time
import mmcv
from torch import Tensor, optim
from mmcv.parallel import scatter
import torch
from .patch_transformer import PatchTransformer
from .patch_applier import PatchApplier
from .nps import NonPrintablityScore
from .patch import init_patch
from .save_img import save_img


def total_variation(image: Tensor):
    """Total variation of an image.

    Args:
        image: 3D Tensor of (N,C,H,W)
    """
    w_axis = (image[:, :, :, 1:] - image[:, :, :, :-1]).abs().sum()
    h_axis = (image[:, :, 1:, :] - image[:, :, :-1, :]).abs().sum()
    return (w_axis + h_axis) / image.numel()

def train_universal_patch(model, data_loader, universal_patch_config, rank, world_size):
    save_path = universal_patch_config.get("save_path", None)
    optimizer = optim.Adam([patch], lr=universal_patch_config['lr'], amsgrad=True)
    num_classes = universal_patch_config.get("num_classes", 80)
    patch_size = universal_patch_config.get("patch_size", 300)
    scale_factor = universal_patch_config.get("scale_factor", 0.2)
    patch_type = universal_patch_config.get("patch_type", "square")


    patch = torch.empty((num_classes, 3, patch_size, patch_size), device=model.device)
    patch.requires_grad_()
    init_patch(patch, universal_patch_config["patch_init"])
    nps_calculator = NonPrintablityScore(universal_patch_config['nps_file']).to(model.device)
    patch_transformer = PatchTransformer(scale=scale_factor, patch_type=patch_type)
    patch_applier = PatchApplier()
    StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.4)
    max_loss = 1e9
    best_patch = None

    if universal_patch_config.get("tensorboard", False):
        from .tensorboard import init_tensorboard
        summary_writer = init_tensorboard(
            osp.join(save_path, "runs")
        )
        print(f"Tensorboard log dir is '{summary_writer.logdir}'", file=sys.stderr)
    else:
        summary_writer = None
    for epoch in range(epoch):
        if rank == 0:
            print(f"\nEpoch: {epoch+1}/{universal_patch_config['epoch']}")
            prog_patch = mmcv.ProgressBar(len(data_loader.dataset))
            time.sleep(2)
        for i, data in enumerate(data_loader):
            optimizer.zero_grad()
            sample = scatter(data, [torch.cuda.current_device()])[0]
            img = sample['img']
            lab = sample['gt_labels']
            target = sample['gt_bboxes'] #list len=batch_size, element shape [num_boxes, 4],type tensor
            target_len = [len(t) for t in target]
            target_len = torch.tensor(target_len, device=img.device)
            target = torch.cat(tuple(target), dim=0)  
            lab = torch.cat(tuple(lab), dim=0)        
            if target.shape[0] == 0:
                continue 
            size = torch.tensor([img.shape[-2], img.shape[-1]]).to(img.device)
            adv_batch, msk_batch = patch_transformer(target, size, lab, patch)
            img_adv = patch_applier(img, target_len, adv_batch, msk_batch)

            sample['img'] = img_adv
            with torch.no_grad():
                result = model(with_patch=True,**sample)
            det_loss = result['loss_obj']
            tv_loss = torch.clamp_min(universal_patch_config["tv_loss"] * total_variation(patch), 0.1)
            nps_loss = nps_calculator(patch)
            loss = det_loss+ tv_loss+universal_patch_config["nps_loss"]*nps_loss

            if rank == 0 and ["tensorboard"]:
                summary_writer.add_scalar("Loss/det_loss", det_loss.item(), i+epoch*len(data_loader))
                summary_writer.add_scalar("Loss/tv_loss", tv_loss.item(), i+epoch*len(data_loader))
                summary_writer.add_scalar("Loss/nps_loss", nps_loss.item(), i+epoch*len(data_loader))
                summary_writer.add_scalar("Loss/total_loss", loss.item(), i+epoch*len(data_loader))
                summary_writer.add_scalar("lr", optimizer.param_groups[0]['lr'], i+epoch*len(data_loader))
                
            loss.backward()
            optimizer.step()
            if rank == 0:
                batch_size = img.shape[0]
                for _ in range(batch_size * world_size):
                    prog_patch.update()
            StepLR.step()
            if loss < max_loss:
                max_loss = loss
                best_patch = patch.detach()
            if rank == 0:
                print('\ndet_loss:', det_loss.item(), 'tv_loss:', tv_loss.item(), 'nps_loss:', nps_loss.item())
        if rank == 0:
            print("\nPatch training finished")
            target = target.split(target_len)
            img_mean = torch.from_numpy(sample['img_metas'][0]['img_norm_cfg']['mean']).to(img.device)
            img_mean = img_mean.unsqueeze(0).unsqueeze(2).unsqueeze(2)
            img_std = torch.from_numpy(sample['img_metas'][0]['img_norm_cfg']['std']).to(img.device)
            img_std = img_std.unsqueeze(0).unsqueeze(2).unsqueeze(2)
            img_transform = (lambda x: (x - img_mean) / img_std, lambda x: x * img_std + img_mean)
            img_adv = img_transform[1](img_adv)
            save_img(img_adv, target, osp.join(save_path, "out/img_adv.png"))
    return best_patch, patch_transformer, patch_applier