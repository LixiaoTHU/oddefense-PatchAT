from mmdet.models import ResNet
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

self = ResNet(depth=50)
self.eval()
inputs = torch.rand(1, 3, 32, 32)
level_outputs = self.forward(inputs)
for level_out in level_outputs:
    print(tuple(level_out.shape))