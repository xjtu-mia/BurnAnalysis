# Implementation of SE blcok in "Squeeze-and-Excitation Networks"
# https://arxiv.org/pdf/1709.01507v4.pdf

import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    """
    NOTE: SE blcok, channel attention
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, _, _ = x.size()
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y.expand_as(x)
