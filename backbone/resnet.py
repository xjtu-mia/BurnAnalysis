import torch.nn as nn
from typing import Union, Type

from .utils import load_state_dict_from_url, log_incompatible_keys


class BasicBlock(nn.Module):
    """BasicBlock for ResNet.
    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int): The ratio of ``out_channels/mid_channels`` where
            ``mid_channels`` is the output channels of conv1. This is a
            reserved argument in BasicBlock and should always be 1. Default: 1.
        stride (int): stride of the block. Default: 1
        dilation (int): dilation of convolution. Default: 1
        downsample (nn.Module, optional): downsample operation on identity
            branch. Default: None.
    """
    expansion = 1
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 dilation=1,
                 downsample=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert out_channels % self.expansion == 0
        self.mid_channels = out_channels // self.expansion
        self.stride = stride
        self.dilation = dilation

        self.conv1 = nn.Conv2d(in_channels,
                               self.mid_channels,
                               3,
                               stride=stride,
                               padding=dilation,
                               dilation=dilation,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.mid_channels)
        self.conv2 = nn.Conv2d(self.mid_channels,
                               out_channels,
                               3,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int): The ratio of ``out_channels/mid_channels`` where
            ``mid_channels`` is the input/output channels of conv2. Default: 4.
        stride (int): stride of the block. Default: 1
        dilation (int): dilation of convolution. Default: 1
        downsample (nn.Module, optional): downsample operation on identity
            branch. Default: None.
    """
    expansion = 4
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 dilation=1,
                 downsample=None,):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert out_channels % self.expansion == 0
        self.mid_channels = out_channels // self.expansion
        self.dilation = dilation

        self.conv1_stride = 1
        self.conv2_stride = stride

        self.conv1 = nn.Conv2d(
            in_channels,
            self.mid_channels,
            1,
            stride=self.conv1_stride,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.mid_channels)
        self.conv2 = nn.Conv2d(
            self.mid_channels,
            self.mid_channels,
            3,
            stride=self.conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(self.mid_channels)
        self.conv3 = nn.Conv2d(
            self.mid_channels,
            out_channels,
            1,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet backbone.

    Refer to the `paper <https://arxiv.org/abs/1512.03385>` for details.

    Args:
        depth (int): Network depth, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Default: 3.
        stem_channels (int): Output channels of the stem layer. Default: 64.
        base_channels (int): Middle channels of the first stage. Default: 64.
        num_stages (int): Stages of the network. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
            Default: ``(1, 2, 2, 2)``.
        dilations (Sequence[int]): Dilation of each stage.
            Default: ``(1, 1, 1, 1)``.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv.
            Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity. Default: True.
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    url_setting = {
        18 : 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth',
        34 : 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet34_8xb32_in1k_20210831-f257d4e6.pth',
        50 : 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
        101 : 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet101_8xb32_in1k_20210831-539c63f8.pth',
        152 : 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet152_8xb32_in1k_20210901-4d7582fa.pth',
    }

    def __init__(self,
                 depth: int,
                 in_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 deep_stem=False):
        super().__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations)
        self.out_names = ['stage1', 'stage2', 'stage3', 'stage4']
        self.deep_stem = deep_stem

        block, stage_blocks = self.arch_settings[depth]

        self._make_stem_layer(in_channels, stem_channels)

        self.in_channels = stem_channels
        self.layer1 = self._make_layer(
            block, stage_blocks[0], base_channels, stride=strides[0], dilation=dilations[0])
        self.layer2 = self._make_layer(
            block, stage_blocks[1], base_channels * 2, stride=strides[1], dilation=dilations[1])
        self.layer3 = self._make_layer(
            block, stage_blocks[2], base_channels * 4, stride=strides[2], dilation=dilations[2])
        self.layer4 = self._make_layer(
            block, stage_blocks[3], base_channels * 8, stride=strides[3], dilation=dilations[3])
    
    def use_pretrained(self, log=True):
        state_dict = load_state_dict_from_url(url=self.url_setting[self.depth])
        state_dict_ = { name.replace('backbone.', '') : weight for name, weight in state_dict.items()}
        incompatible = self.load_state_dict(state_dict_, strict=False)
        if log:
            log_incompatible_keys(incompatible)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            # 为了match预训练权重中的stem, 这里封装了一个Conv3x3BNReLU模块
            self.stem = nn.Sequential(
                Conv3x3BNReLU(in_channels, stem_channels // 2),
                Conv3x3BNReLU(stem_channels // 2, stem_channels // 2),
                Conv3x3BNReLU(stem_channels // 2, stem_channels),
                       )              
        else:
            self.conv1 = nn.Conv2d(
                in_channels, stem_channels, 7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(stem_channels)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        blocks: int,
        channels: int,
        stride: int = 1,
        dilation: int = 1,
    ) -> nn.Sequential:
        
        downsample = None
        # 每个stage的第一个残差块降采样
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

    def forward(self, x):
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        # 残差层
        outs = []
        x = self.layer1(x)
        outs.append(x)
        x = self.layer2(x)
        outs.append(x)
        x = self.layer3(x)
        outs.append(x)
        x = self.layer4(x)
        outs.append(x)
        return {name: feats for name, feats in zip(self.out_names, outs)}

class Conv3x3BNReLU(nn.Module):
    def __init__(self, in_chs, chs):
        super().__init__()
        self.conv = nn.Conv2d(in_chs, chs, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(chs)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResNetV1c(ResNet):
    """ResNetV1c backbone.

    This variant is described in `Bag of Tricks.
    <https://arxiv.org/pdf/1812.01187.pdf>`_.

    Compared with default ResNet(ResNetV1b), ResNetV1c replaces the 7x7 conv
    in the input stem with three 3x3 convs.
    """
    url_setting = {
        50 : 'https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1c50_8xb32_in1k_20220214-3343eccd.pth',
        101 : 'https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1c101_8xb32_in1k_20220214-434fe45f.pth',
        152 : 'https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1c152_8xb32_in1k_20220214-c013291f.pth',
    }
    def __init__(self, **kwargs):
        depth = kwargs.get('depth')
        assert depth in [50, 101, 152], f"ResNetV1c-{depth} is not surpported. The surpported depthes are [50, 101, 152]."
        super().__init__(
            deep_stem=True, **kwargs)
        