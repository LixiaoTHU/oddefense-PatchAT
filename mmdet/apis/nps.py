import torch
from torch import Tensor
from torch.nn import Module


def non_printablity_score(image: Tensor, color_array: Tensor):
    """Compute non-printablity of image regarding color array.

    Args:
        image: 4D Tensor of (num_class,C,H,W)
        color_array: 4D Tensor of (N,C,1,1)
    """
    img = image.unsqueeze(1)
    img = img.expand(-1, color_array.shape[0], -1, -1, -1)
    color_array = color_array.unsqueeze(0)
    color_array = color_array.expand(image.shape[0], -1, -1, -1, -1)
    color_dist = (img - color_array).square().sum(dim=2).sqrt()
    color_dist_prod = torch.min(color_dist, dim=1)[0] #num_class,H,W
    nps_score = torch.sum(color_dist_prod)
    return nps_score / image.numel()


class NonPrintablityScore(Module):
    color_array: Tensor

    def __init__(self, color_file: str) -> None:
        super().__init__()
        self.register_buffer("color_array", self.get_color_array(color_file))

    @classmethod
    def get_color_array(cls, color_file: str):
        with open(color_file, "r") as f:
            colors = [tuple(map(float, _.rstrip().split(","))) for _ in f]
        colors = torch.tensor(colors)
        return colors.reshape(*colors.shape, 1, 1)

    def forward(self, x: Tensor):
        return non_printablity_score(x, self.color_array)
