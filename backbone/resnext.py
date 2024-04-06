from typing import Type, Union
import torch.nn as nn

from .resnet import Bottleneck as _Bottleneck
from .resnet import ResNet


class Bottleneck(_Bottleneck):
    """Bottleneck block for ResNeXt.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        groups (int): Groups of conv2.
        width_per_group (int): Width per group of conv2. 64x4d indicates
            ``groups=64, width_per_group=4`` and 32x8d indicates
            ``groups=32, width_per_group=8``.
        stride (int): stride of the block. Default: 1
        dilation (int): dilation of convolution. Default: 1
        downsample (nn.Module, optional): downsample operation on identity
            branch. Default: None
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 base_channels=64,
                 groups=32,
                 width_per_group=4,
                 **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)
        self.groups = groups
        self.width_per_group = width_per_group

        # For ResNet bottleneck, middle channels are determined by expansion
        # and out_channels, but for ResNeXt bottleneck, it is determined by
        # groups and width_per_group and the stage it is located in.
        if groups != 1:
            assert self.mid_channels % base_channels == 0
            self.mid_channels = (
                groups * width_per_group * self.mid_channels // base_channels)
        
        self.conv1 = nn.Conv2d(self.in_channels, self.mid_channels, 1, stride=self.conv1_stride, bias=False)
        self.bn1 = nn.BatchNorm2d(self.mid_channels)
        
        self.conv2 = nn.Conv2d(self.mid_channels, self.mid_channels, 3, self.conv2_stride, 
                               padding=self.dilation, dilation=self.dilation,groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(self.mid_channels)

        self.conv3 = nn.Conv2d(self.mid_channels, self.out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.out_channels)

class ResNeXt(ResNet):
    """ResNeXt backbone.

    Refer to the `paper <https://arxiv.org/abs/1611.05431>` for details.

    Args:
        depth (int): Network depth, from {50, 101, 152}.
        groups (int): Groups of conv2 in Bottleneck. Default: 32.
        width_per_group (int): Width per group of conv2 in Bottleneck.
            Default: 4.
        in_channels (int): Number of input image channels. Default: 3.
        stem_channels (int): Output channels of the stem layer. Default: 64.
        num_stages (int): Stages of the network. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
            Default: ``(1, 2, 2, 2)``.
        dilations (Sequence[int]): Dilation of each stage.
            Default: ``(1, 1, 1, 1)``.
    """
    arch_settings = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    url_setting = {
        50 : 'https://download.openmmlab.com/mmclassification/v0/resnext/resnext50_32x4d_b32x8_imagenet_20210429-56066e27.pth',
        101 : 'https://download.openmmlab.com/mmclassification/v0/resnext/resnext101_32x4d_b32x8_imagenet_20210506-e0fa3dd5.pth',
        152 : 'https://download.openmmlab.com/mmclassification/v0/resnext/resnext152_32x4d_b32x8_imagenet_20210524-927787be.pth',
    }
    def __init__(self, depth: int, groups=32, width_per_group=4, **kwargs):
        self.groups = groups
        self.width_per_group = width_per_group
        super(ResNeXt, self).__init__(depth, **kwargs)
    
    def _make_layer(
            self,
            block: Bottleneck,
            blocks: int,
            channels: int,
            stride: int = 1,
            dilation: int = 1,
        ) -> nn.Sequential:
            
        downsample = None
        # 每个stage的首个残差块降采样
        if self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels *
                            block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.in_channels,
                channels * block.expansion,
                groups = self.groups,
                width_per_group = self.width_per_group,
                stride=stride,
                dilation=1,
                downsample=downsample)
        )
        self.in_channels = channels * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_channels,
                    channels * block.expansion,
                    dilation=dilation,
                )
            )
        return nn.Sequential(*layers)