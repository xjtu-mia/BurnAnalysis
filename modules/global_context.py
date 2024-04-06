# Implementation of: "GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond"
# https://arxiv.org/abs/1904.11492v1.pdf


import torch
import torch.nn as nn


class GCBlock(nn.Module):
    """GlobalContext module in GCNet.

    Args:
        in_channels (int): Channels of the input feature map.
        ratio (float): Ratio of channels of transform bottleneck
        pooling_type (str): Pooling method for context modeling.
            Options are 'att' and 'avg', stand for attention pooling and
            average pooling respectively. Default: 'att'.
        fusion_types (Sequence[str]): Fusion method for feature fusion,
            Options are 'channels_add', 'channel_mul', stand for channelwise
            addition and multiplication respectively. Default: ('channel_add',)
    """

    def __init__(self,
                 in_channels: int,
                 reduction_ratio: int = 4,
                 pooling_type: str = 'att',
                 fusion_types: tuple = ('channel_add', 'channel_mul')):
        super().__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.in_channels = in_channels
        self.planes = in_channels // reduction_ratio
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(in_channels, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.in_channels, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.in_channels, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.in_channels, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.in_channels, kernel_size=1))
        else:
            self.channel_mul_conv = None
        

    def spatial_pool(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, 1, C, H * W]
            input_x = input_x.view(N, C, -1).unsqueeze(1)
            # [N, 1, H * W]
            context_mask = self.conv_mask(x).view(N, 1, -1)
            # [N, 1, H * W, 1]
            context_mask = self.softmax(context_mask).unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(N, C, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)
        return context

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        return out
    