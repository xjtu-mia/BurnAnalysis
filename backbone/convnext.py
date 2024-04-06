import os
from typing import Sequence
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import load_state_dict_from_url, log_incompatible_keys


def drop_path(x: torch.Tensor,
              drop_prob: float = 0.,
              training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py  # noqa: E501
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # handle tensors with different dimensions, not just 4D tensors.
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
    output = x.div(keep_prob) * random_tensor.floor()
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py  # noqa: E501

    Args:
        drop_prob (float): Probability of the path to be zeroed. Default: 0.1
    """

    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)
    

class LayerNorm2d(nn.LayerNorm):
    """LayerNorm on channels for 2d images.

    Args:
        num_channels (int): The number of channels of the input tensor.
        eps (float): a value added to the denominator for numerical stability.
            Defaults to 1e-5.
        elementwise_affine (bool): a boolean value that when set to ``True``,
            this module has learnable per-element affine parameters initialized
            to ones (for weights) and zeros (for biases). Defaults to True.
    """

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)
        self.num_channels = self.normalized_shape[0]

    def forward(self, x, data_format='channel_first'):
        """Forward method.

        Args:
            x (torch.Tensor): The input tensor.
            data_format (str): The format of the input tensor. If
                ``"channel_first"``, the shape of the input tensor should be
                (B, C, H, W). If ``"channel_last"``, the shape of the input
                tensor should be (B, H, W, C). Defaults to "channel_first".
        """
        assert x.dim() == 4, 'LayerNorm2d only supports inputs with shape ' \
            f'(N, C, H, W), but got tensor with shape {x.shape}'
        if data_format == 'channel_last':
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias,
                             self.eps)
        elif data_format == 'channel_first':
            x = x.permute(0, 2, 3, 1)
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias,
                             self.eps)
            # If the output is discontiguous, it may cause some unexpected
            # problem in the downstream tasks
            x = x.permute(0, 3, 1, 2).contiguous()
        return x


class GRN(nn.Module):
    """Global Response Normalization Module.

    Come from `ConvNeXt V2: Co-designing and Scaling ConvNets with Masked
    Autoencoders <http://arxiv.org/abs/2301.00808>`_

    Args:
        in_channels (int): The number of channels of the input tensor.
        eps (float): a value added to the denominator for numerical stability.
            Defaults to 1e-6.
    """

    def __init__(self, in_channels, eps=1e-6):
        super().__init__()
        self.in_channels = in_channels
        self.gamma = nn.Parameter(torch.zeros(in_channels))
        self.beta = nn.Parameter(torch.zeros(in_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor, data_format='channel_first'):
        """Forward method.

        Args:
            x (torch.Tensor): The input tensor.
            data_format (str): The format of the input tensor. If
                ``"channel_first"``, the shape of the input tensor should be
                (B, C, H, W). If ``"channel_last"``, the shape of the input
                tensor should be (B, H, W, C). Defaults to "channel_first".
        """
        if data_format == 'channel_last':
            gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
            nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
            x = self.gamma * (x * nx) + self.beta + x
        elif data_format == 'channel_first':
            gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
            x = self.gamma.view(1, -1, 1, 1) * (x * nx) + self.beta.view(
                1, -1, 1, 1) + x
        return x


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block.

    Args:
        in_channels (int): The number of input channels.
        dw_conv_cfg (dict): Config of depthwise convolution.
            Defaults to ``dict(kernel_size=7, padding=3)``.
        mlp_ratio (float): The expansion ratio in both pointwise convolution.
            Defaults to 4.
        linear_pw_conv (bool): Whether to use linear layer to do pointwise
            convolution. More details can be found in the note.
            Defaults to True.

    Note:
        There are two equivalent implementations:

        1. DwConv -> LayerNorm -> 1x1 Conv -> GELU -> 1x1 Conv;
           all outputs are in (N, C, H, W).
        2. DwConv -> LayerNorm -> Permute to (N, H, W, C) -> Linear -> GELU
           -> Linear; Permute back

        As default, we use the second to align with the official repository.
        And it may be slightly faster.
    """

    def __init__(self,
                 in_channels,
                 dw_conv_cfg=dict(kernel_size=7, padding=3),
                 mlp_ratio=4.,
                 linear_pw_conv=True,
                 use_grn=False,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,):
        super().__init__()

        self.depthwise_conv = nn.Conv2d(
            in_channels, in_channels, groups=in_channels, **dw_conv_cfg)

        self.linear_pw_conv = linear_pw_conv
        self.norm = LayerNorm2d(in_channels)

        mid_channels = int(mlp_ratio * in_channels)
        if self.linear_pw_conv:
            # Use linear layer to do pointwise conv.
            pw_conv = nn.Linear
        else:
            pw_conv = partial(nn.Conv2d, kernel_size=1)

        self.pointwise_conv1 = pw_conv(in_channels, mid_channels)
        self.act = nn.GELU()
        self.pointwise_conv2 = pw_conv(mid_channels, in_channels)

        if use_grn:
            self.grn = GRN(mid_channels)
        else:
            self.grn = None

        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((in_channels)),
            requires_grad=True) if layer_scale_init_value > 0. else None
        
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.depthwise_conv(x)

        if self.linear_pw_conv:
            x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
            x = self.norm(x, data_format='channel_last')
            x = self.pointwise_conv1(x)
            x = self.act(x)
            if self.grn is not None:
                x = self.grn(x, data_format='channel_last')
            x = self.pointwise_conv2(x)
            x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        else:
            x = self.norm(x, data_format='channel_first')
            x = self.pointwise_conv1(x)
            x = self.act(x)

            if self.grn is not None:
                x = self.grn(x, data_format='channel_first')
            x = self.pointwise_conv2(x)

        if self.gamma is not None:
            x = x.mul(self.gamma.view(1, -1, 1, 1))

        x = shortcut + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    """ConvNeXt v1&v2 backbone.

    A PyTorch implementation of `A ConvNet for the 2020s
    <https://arxiv.org/abs/2201.03545>`_ and
    `ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders
    <http://arxiv.org/abs/2301.00808>`_

    Modified from the `official repo
    <https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py>`_
    and `timm
    <https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/convnext.py>`_.

    To use ConvNeXt v2, please set ``use_grn=True`` and ``layer_scale_init_value=0.``.

    Args:
        arch (str | dict): The model's architecture. If string, it should be
            one of architecture in ``ConvNeXt.arch_settings``. And if dict, it
            should include the following two keys:

            - depths (list[int]): Number of blocks at each stage.
            - channels (list[int]): The number of channels at each stage.

            Defaults to 'tiny'.
        in_channels (int): Number of input image channels. Defaults to 3.
        stem_patch_size (int): The size of one patch in the stem layer.
            Defaults to 4.
        linear_pw_conv (bool): Whether to use linear layer to do pointwise
            convolution. Defaults to True.
        drop_path_rate(float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value(float): Init value for Layer Scale.
            Defaults to 1e-6.
        out_indices(tuple[int]): Output from which stages.
            Defaults to (0, 1, 2, 3), means all stages.
        use_grn (bool): Whether to add Global Response Normalization in the
            blocks. Defaults to False.
    """  
    arch_settings = {
        'atto': {
            'depths': [2, 2, 6, 2],
            'channels': [40, 80, 160, 320]
        },
        'femto': {
            'depths': [2, 2, 6, 2],
            'channels': [48, 96, 192, 384]
        },
        'pico': {
            'depths': [2, 2, 6, 2],
            'channels': [64, 128, 256, 512]
        },
        'nano': {
            'depths': [2, 2, 8, 2],
            'channels': [80, 160, 320, 640]
        },
        **dict.fromkeys(
            ['t', 'tiny'],
            {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
            }
        ),
        **dict.fromkeys(
            ['s', 'small'],
            {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
            }
        ),
        **dict.fromkeys(
            ['b', 'base'],
            {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
            }
        ),
    }

    url_setting = {
        # 'atto', 'femto', 'pico', 'nano' convnext v2
        'atto': 'https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-atto_fcmae-pre_3rdparty_in1k_20230104-23765f83.pth',
        'femto': 'https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-femto_fcmae-pre_3rdparty_in1k_20230104-92a75d75.pth',
        'pico': 'https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-pico_fcmae-pre_3rdparty_in1k_20230104-d20263ca.pth',
        'nano': 'https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-nano_fcmae-pre_3rdparty_in1k_20230104-fe1aaaf2.pth',
        **dict.fromkeys(
            ['t', 'tiny'],
            'https://download.openmmlab.com/mmclassification/v0/convnext/convnext-tiny_32xb128-noema_in1k_20221208-5d4509c7.pth'
        ),
        **dict.fromkeys(
            ['s', 'small'],
            'https://download.openmmlab.com/mmclassification/v0/convnext/convnext-small_32xb128_in1k_20221207-4ab7052c.pth',
        ),
        **dict.fromkeys(
            ['b', 'base'],
            'https://download.openmmlab.com/mmclassification/v0/convnext/convnext-base_32xb128_in1k_20221207-fbdb5eb9.pth',
        )
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 linear_pw_conv=True,
                 drop_path_rate=0.5,
                 layer_scale_init_value=1e-6,
                 out_indices=(0, 1, 2, 3),
                 use_grn=False,):
        super().__init__()

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch_setting = self.arch_settings[arch]
        self.arch = arch
        self.depths = arch_setting['depths']
        self.channels = arch_setting['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)
        self.out_indices = out_indices
        self.out_names = [f"stage{i + 1}"for i in out_indices]

        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            LayerNorm2d(self.channels[0]),
        )
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    LayerNorm2d(self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            stage = nn.Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    linear_pw_conv=linear_pw_conv,
                    use_grn=use_grn,
                    drop_path_rate=dpr[block_idx + j],
                    layer_scale_init_value=layer_scale_init_value) for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

        for i in out_indices:
            norm_layer = LayerNorm2d(self.channels[i])
            self.add_module(f'norm{i}', norm_layer)

    def use_pretrained(self, log=True):
        # 使用的预训练模型中使用了mmengine中定义的日志类，未安装加载模型会出错
        try: 
            import mmengine
        except ImportError:
            os.system('pip install mmengine==0.9.0')
        state_dict = load_state_dict_from_url(url=self.url_setting[self.arch])
        state_dict_ = { name.replace('backbone.', '') : weight for name, weight in state_dict.items()}
        incompatible = self.load_state_dict(state_dict_, strict=False)
        if log:
            log_incompatible_keys(incompatible)

    def forward(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                outs.append(norm_layer(x))
        return {name: feats for name, feats in zip(self.out_names, outs)}
