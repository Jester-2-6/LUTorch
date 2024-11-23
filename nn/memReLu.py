import torch
import torch.nn as nn


class memReLu(nn.ReLU):
    def __init__(self):
        super(memReLu, self).__init__()

    def forward(self, x):
        return torch.clamp(x, 0, 1)
