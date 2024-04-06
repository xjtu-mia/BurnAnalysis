# Dilated Sub-pixel Convolution Pp-sampling blcok

import torch
import torch.nn as nn
from torch import Tensor
from typing import List


class DSCUBlcok(nn.Module):
    def __init__(self,
                 in_channels: int,
                 channels: int,
                 dilations: List[int] = [1, 3, 6, 12]):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.dilations = dilations
        self.aspp = ASPPModule(dilations, in_channels, channels)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self, x):
        out = self.aspp(x)
        out = torch.cat(out, dim=1)
        out = self.pixel_shuffle(out)
        return out

class ConvModule(nn.Module):
    "Conv->Norm->Relu"
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 use_norm: bool = True,
                 use_act: bool = True,
                 inplace: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2 if dilation == 1 else dilation,
            dilation=dilation,
            groups=groups,
            bias=False if use_norm else True,
        )
        self.use_norm = use_norm
        self.use_act = use_act
        if use_norm:
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = None
        if use_act:
            self.act = nn.ReLU(inplace=inplace)
        else:
            self.act = None

    def forward(self, x: Tensor):
        x = self.conv(x)
        if self.use_norm:
            x = self.norm(x)
        if self.use_act:
            x = self.act(x)
        return x

class ASPPModule(nn.ModuleList):
    """Atrous Spatial Pyramid Pooling (ASPP) Module.

    Args:
        dilations (tuple[int]): Dilation rate of each layer.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
    """

    def __init__(self, dilations, in_channels, channels):
        super().__init__()
        self.dilations = dilations
        self.in_channels = in_channels
        self.channels = channels
        for dilation in dilations:
            self.append(
                ConvModule(
                    self.in_channels,
                    self.channels,
                    kernel_size=1 if dilation == 1 else 3,
                    dilation=dilation,
                    use_norm=True,
                    use_act=True,
                    inplace=False)
            )

    def forward(self, x):
        aspp_outs = []
        for aspp_module in self:
            aspp_outs.append(aspp_module(x))
        return aspp_outs