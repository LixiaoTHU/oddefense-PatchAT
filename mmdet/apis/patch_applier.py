import torch
from torch import Tensor
from torch.nn import Module


class PatchApplier(Module):
    """Applies adversarial patches to images.

    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        img_batch: Tensor,
        target_lens: Tensor,
        adv_batch: Tensor,
        msk_batch: Tensor,
    ):
        cumsum = 0
        for i in range(len(img_batch)):
            l = target_lens[i]
            for j in range(cumsum, cumsum + l):
                img_batch[i] = torch.where(
                    msk_batch[j] < 0.5, img_batch[i], adv_batch[j]
                )
            cumsum += l
        return img_batch
