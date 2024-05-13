from typing import Optional

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch import Tensor

from .typing import _pair, _size_2_t


def init_patch(patch: Tensor, patch_init: float | int | str):
    """In-place init of patch."""
    if isinstance(patch_init, float):
        patch.fill_(patch_init)
    elif isinstance(patch_init, int):
        patch.fill_(patch_init / 255)
    elif isinstance(patch_init, str):
        if patch_init == "random":
            patch.uniform_()
        else:
            raise ValueError(patch_init)
    else:
        raise ValueError(patch_init)


def load_patch(fn: str, patch_size: Optional[_size_2_t] = None):
    """Load patch from file."""
    if fn.endswith((".npy", ".npz")):
        patch = torch.from_numpy(np.load(fn))
    elif fn.endswith((".pt", ".pth")):
        patch: Tensor = torch.load(fn)["patch"]
    else:
        patch = F.to_tensor(Image.open(fn).convert("RGB"))
    if patch_size is not None:
        patch_size = _pair(patch_size)
        patch = F.resize(patch, patch_size)
    return patch
