import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module


from .median_pool import MedianPool2d
from .bbox_utils import bbox_from_x1y1x2y2
import cv2

# Tolerance on comparison
_TOL = 1e-8


class PatchTransformer(Module):
    """PatchTransformer: transforms batch of patches

    Module providing the functionality necessary to transform a batch of patches, randomly adjusting brightness and
    contrast, adding random amount of noise, and rotating randomly. Resizes patches according to as size based on the
    batch of labels, and pads them to the dimension of an image.

    """

    kernel: Tensor

    def __init__(
        self,
        contrast: tuple[float, float] = (0.8, 1.2),
        brightness: tuple[float, float] = (-0.1, 0.1),
        angle: tuple[float, float] = (-20.0, 20.0),
        noise_factor=0.1,
        do_rotate=False,
        rand_loc=True,
        lc_scale=0.1,
        pooling="median",
        rand_sub=False,
        old_fasion=False,
        scale=0.2,
        y_ratio=1.0,
        patch_type = "square",
    ):
        super().__init__()
        self.min_contrast, self.max_contrast = contrast
        self.min_brightness, self.max_brightness = brightness
        self.noise_factor = noise_factor
        self.minangle, self.maxangle = map(self.degree_to_radius, angle)
        self.medianpooler = MedianPool2d(7, same=True)
        self.do_rotate = do_rotate
        self.rand_loc = rand_loc
        self.lc_scale = lc_scale
        assert pooling in ("median", "avg", "gauss", "none")
        self.pooling = pooling
        self.rand_sub = rand_sub
        self.old_fasion = old_fasion
        self.scale = scale
        self.y_ratio = y_ratio
        self.patch_type = patch_type

        # 5*5 gaussian kernel
        ksize = 5
        half = (ksize - 1) * 0.5
        sigma = 0.3 * (half - 1) + 0.8
        x = np.arange(-half, half + 1)
        x = np.exp(-np.square(x / sigma) / 2)
        x = np.outer(x, x)
        x = x / x.sum()
        x = torch.from_numpy(x).float()
        kernel = torch.zeros(3, 3, ksize, ksize)
        for i in range(3):
            kernel[i, i] = x
        self.register_buffer("kernel", kernel)
        """
        kernel = torch.cuda.FloatTensor([[0.003765, 0.015019, 0.023792, 0.015019, 0.003765],                                                                                    
                                         [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],                                                                                    
                                         [0.023792, 0.094907, 0.150342, 0.094907, 0.023792],                                                                                    
                                         [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],                                                                                    
                                         [0.003765, 0.015019, 0.023792, 0.015019, 0.003765]])
        self.kernel = kernel.unsqueeze(0).unsqueeze(0).expand(3,3,-1,-1)
        # It's wrong!
        """

    @staticmethod
    def degree_to_radius(x: int | float):
        return x / 180 * math.pi

    @staticmethod
    def radius_to_degree(x: int | float):
        return x / math.pi * 180

    def to_fixed_transform(self):
        contrast = (self.min_contrast + self.max_contrast) / 2
        brightness = (self.min_brightness + self.max_brightness) / 2
        angle = self.radius_to_degree((self.minangle + self.maxangle) / 2)
        return PatchTransformer(
            (contrast, contrast),
            (brightness, brightness),
            (angle, angle),
            0.0,
            self.do_rotate,
            False,
            self.lc_scale,
            self.pooling,
            False,
            self.old_fasion,
            self.scale,
            self.y_ratio,
        )

    def forward(
        self,
        lab_batch: Tensor, #cxcywh
        img_size: Tensor,
        lab: Tensor,
        adv_patch: Tensor = None,
    ):
        if adv_patch is None:
            return self.forward_bbox_patch(lab_batch, img_size, lab)
        else:
            return self.forward_muti_class_patch(lab_batch, img_size, lab, adv_patch)

    def forward_muti_class_patch(
        self,
        lab_batch: Tensor, #cxcywh
        img_size: Tensor,
        lab: Tensor,
        adv_patch: Tensor = None,
    ):
        lab_batch = bbox_from_x1y1x2y2(lab_batch, "cxcywh")
        # Followings compute on cxcywh format bouding box
        SBS, _ = lab_batch.shape
        _, C, H, W = adv_patch.shape

        # Make a batch of patches
        adv_batch = adv_patch[lab,:,:,:]
        #adv_batch = adv_patch.expand(SBS, -1, -1, -1)

        # Where the label class_id is 1 we don't want a patch (padding) --> fill mask with zero's
        if self.patch_type == "round":
            x, y = torch.meshgrid([torch.arange(0, adv_batch.shape[-2]), torch.arange(0, adv_batch.shape[-1])])
            x_ = torch.tensor(adv_batch.shape[-2]/2)
            y_ = torch.tensor(adv_batch.shape[-1]/2)
            r = torch.tensor(adv_batch.shape[-2]/2)
            msk_batch = (torch.sqrt((x - x_)**2 + (y - y_)**2) <= r).to(adv_patch.device)
            msk_batch = msk_batch.unsqueeze(0).unsqueeze(0)
            msk_batch = msk_batch.expand(SBS, C, -1, -1)
        else:
            msk_batch = adv_patch.new_ones(adv_batch.shape, dtype=torch.bool)

        # Resizes and rotates
        # Relative scale is 0.2
        target_x = lab_batch[:, 0]
        target_y = lab_batch[:, 1]
        targetoff_x = lab_batch[:, 2]
        targetoff_y = lab_batch[:, 3]
        scale = self.scale * torch.sqrt(targetoff_x.square() + targetoff_y.square())
        if self.rand_loc:
            # Randomly move the patch
            off_x = targetoff_x * (
                adv_patch.new_empty(targetoff_x.size()).uniform_(
                    -self.lc_scale, self.lc_scale
                )
            )
            target_x = target_x + off_x
            off_y = targetoff_y * (
                adv_patch.new_empty(targetoff_y.size()).uniform_(
                    -self.lc_scale, self.lc_scale
                )
            )
            target_y = target_y + off_y
        
        tx = -(target_x-img_size[1]/2)*2
        ty = -(target_y-img_size[0]/2)*2


        theta = adv_patch.new_zeros(SBS, 2, 3)
        theta[:, 0, 0] = img_size[1]/scale
        theta[:, 0, 1] = 0
        theta[:, 0, 2] = tx/scale
        theta[:, 1, 0] = 0
        theta[:, 1, 1] = img_size[0]/scale
        theta[:, 1, 2] = ty/scale
        
        # theta = theta / scale.reshape(-1, 1, 1)
        # Affine transform to image space
        grid = F.affine_grid(
            theta, [SBS, C, img_size[0], img_size[1]], align_corners=False
            )

        
        adv_batch_t = F.grid_sample(adv_batch, grid, align_corners=False)
        msk_batch_t = F.grid_sample(msk_batch.float(), grid, align_corners=False)
        adv_batch_t = torch.clamp(adv_batch_t, 0, 1)
        
        return adv_batch_t, msk_batch_t


    def forward_bbox_patch(
            self,
            lab_batch: Tensor, #cxcywh
            img_size: Tensor,
            lab: Tensor,
        ):
        lab_batch = bbox_from_x1y1x2y2(lab_batch, "cxcywh")
        SBS, _ = lab_batch.shape
        adv_patch = torch.zeros(3, 300, 300, device=lab.device)
        adv_patch = adv_patch.unsqueeze(0)
        adv_batch = adv_patch.expand(SBS, -1, -1, -1)

        if self.patch_type == "round":
            x, y = torch.meshgrid([torch.arange(0, adv_batch.shape[-2]), torch.arange(0, adv_batch.shape[-1])])
            x_ = torch.tensor(adv_batch.shape[-2]/2)
            y_ = torch.tensor(adv_batch.shape[-1]/2)
            r = torch.tensor(adv_batch.shape[-2]/2)
            msk_batch = (torch.sqrt((x - x_)**2 + (y - y_)**2) <= r).to(lab.device)
            msk_batch = msk_batch.unsqueeze(0).unsqueeze(0)
            msk_batch = msk_batch.expand(SBS, 3, -1, -1)
        else:
            msk_batch = adv_batch.new_ones(adv_batch.shape, dtype=torch.bool)
        
        target_x = lab_batch[:, 0]
        target_y = lab_batch[:, 1]
        targetoff_x = lab_batch[:, 2]
        targetoff_y = lab_batch[:, 3]
        scale = self.scale * torch.sqrt(targetoff_x.square() + targetoff_y.square())
        if self.rand_loc:
            # Randomly move the patch
            off_x = targetoff_x * (
                    adv_patch.new_empty(targetoff_x.size()).uniform_(
                        -self.lc_scale, self.lc_scale
                    )
                )
            target_x = target_x + off_x
            off_y = targetoff_y * (
                    adv_patch.new_empty(targetoff_y.size()).uniform_(
                        -self.lc_scale, self.lc_scale
                    )
                )
            target_y = target_y + off_y
            
        tx = -(target_x-img_size[1]/2)*2
        ty = -(target_y-img_size[0]/2)*2


        theta = adv_patch.new_zeros(SBS, 2, 3)
        theta[:, 0, 0] = img_size[1]/scale
        theta[:, 0, 1] = 0
        theta[:, 0, 2] = tx/scale
        theta[:, 1, 0] = 0
        theta[:, 1, 1] = img_size[0]/scale
        theta[:, 1, 2] = ty/scale
            
        # theta = theta / scale.reshape(-1, 1, 1)
        # Affine transform to image space
        grid = F.affine_grid(
            theta, [SBS, 3, img_size[0], img_size[1]], align_corners=False
            )

            
        adv_batch_t = F.grid_sample(adv_batch, grid, align_corners=False)
        msk_batch_t = F.grid_sample(msk_batch.float(), grid, align_corners=False)
        adv_batch_t = torch.clamp(adv_batch_t, 0, 1)
        
        msk_batch_t = msk_batch_t.bool()

        msk = torch.any(msk_batch_t, dim = 0)
        msk = msk.int()

        return msk
