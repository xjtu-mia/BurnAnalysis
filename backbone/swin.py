
import logging
import torch
import torch.nn as nn

from .transformer import FFN, ShiftWindowMSA, PatchMerging, PatchEmbed
from .utils import log_incompatible_keys, load_state_dict_from_url


class SwinBlock(nn.Module):
    """Swin Transformer block.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window. Defaults to 7.
        shift (bool): Shift the attention window or not. Defaults to False.
        ffn_ratio (float): The expansion ratio of feedforward network hidden
            layer channels. Defaults to 4.
        pad_small_map (bool): If True, pad the small feature map to the window
            size, which is common used in detection and segmentation. If False,
            avoid shifting window and shrink the window size to the size of
            feature map, which is common used in classification.
            Defaults to False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size=7,
                 shift=False,
                 ffn_ratio=4.,
                 pad_small_map=False):

        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dims)
        self.attn = ShiftWindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            pad_small_map=pad_small_map
        )

        self.norm2 = nn.LayerNorm(embed_dims)

        self.ffn = FFN(embed_dims=embed_dims, feedforward_channels=int(embed_dims*ffn_ratio), num_fcs=2)

    def forward(self, x, hw_shape):

        identity = x
        x = self.norm1(x)
        x = self.attn(x, hw_shape)
        x = x + identity

        identity = x
        x = self.norm2(x)
        x = self.ffn(x, identity=identity)

        return x


class SwinBlockSequence(nn.Module):
    """Module with successive Swin Transformer blocks and downsample layer.

    Args:
        embed_dims (int): Number of input channels.
        depth (int): Number of successive swin transformer blocks.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window. Defaults to 7.
        downsample (bool): Downsample the output of blocks by patch merging.
            Defaults to False.
        block_cfgs (Sequence[dict] | dict): The extra config of each block.
            Defaults to empty dicts.
        pad_small_map (bool): If True, pad the small feature map to the window
            size, which is common used in detection and segmentation. If False,
            avoid shifting window and shrink the window size to the size of
            feature map, which is common used in classification.
            Defaults to False.
    """

    def __init__(self,
                 embed_dims,
                 depth,
                 num_heads,
                 window_size=7,
                 downsample=False,
                 pad_small_map=False):
        super().__init__()

        self.embed_dims = embed_dims
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = SwinBlock(embed_dims, num_heads, window_size, 
                              shift=False if i % 2 == 0 else True,
                              pad_small_map=pad_small_map)
            self.blocks.append(block)

        if downsample:
            self.downsample = PatchMerging(in_channels=embed_dims, out_channels=2 * embed_dims)
        else:
            self.downsample = None

    def forward(self, x, in_shape, do_downsample=True):
        for block in self.blocks:
            x = block(x, in_shape)

        if self.downsample is not None and do_downsample:
            x, out_shape = self.downsample(x, in_shape)
        else:
            out_shape = in_shape
        return x, out_shape

    @property
    def out_channels(self):
        if self.downsample:
            return self.downsample.out_channels
        else:
            return self.embed_dims


class SwinTransformer(nn.Module):

    """Swin Transformer backbone.
    Refer to the `paper <https://arxiv.org/abs/2103.14030>` for details.

    Args:
        arch (str): Swin Transformer architecture. If use string, choose
            from 'tiny', 'small', 'base' and 'large'. Defaults to 'tiny'.
        img_size (int | tuple): The expected input image shape. Because we
            support dynamic input shape, just set the argument to the most
            common input image shape. Defaults to 224.
        patch_size (int | tuple): The patch size in patch embedding.
            Defaults to 4.
        in_channels (int): The num of input channels. Defaults to 3.
        window_size (int): The height and width of the window. Defaults to 7.
        drop_rate (float): Dropout rate after embedding. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.1.
        out_after_downsample (bool): Whether to output the feature map of a
            stage after the following downsample layer. Defaults to False.
        interpolate_mode (str): Select the interpolate mode for absolute
            position embeding vector resize. Defaults to "bicubic".
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Defaults to False.
        pad_small_map (bool): If True, pad the small feature map to the window
            size, which is common used in detection and segmentation. If False,
            avoid shifting window and shrink the window size to the size of
            feature map, which is common used in classification.
            Defaults to False.
    """
    arch_zoo = {
        **dict.fromkeys(['t', 'tiny'],
                        {'embed_dims': 96,
                         'depths':     [2, 2,  6,  2],
                         'num_heads':  [3, 6, 12, 24]}),
        **dict.fromkeys(['s', 'small'],
                        {'embed_dims': 96,
                         'depths':     [2, 2, 18,  2],
                         'num_heads':  [3, 6, 12, 24]}),
        **dict.fromkeys(['b', 'base'],
                        {'embed_dims': 128,
                         'depths':     [2, 2, 18,  2],
                         'num_heads':  [4, 8, 16, 32]}),
        **dict.fromkeys(['l', 'large'],
                        {'embed_dims': 192,
                         'depths':     [2,  2, 18,  2],
                         'num_heads':  [6, 12, 24, 48]}),
    }  

    url_setting = {
        **dict.fromkeys(
            ['t', 'tiny'],
            'https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth'
        ),
        **dict.fromkeys(
            ['s', 'small'],
            'https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_small_224_b16x64_300e_imagenet_20210615_110219-7f9d988b.pth'
        ),
        **dict.fromkeys(
            ['b', 'base'],
            'https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_base_224_b16x64_300e_imagenet_20210616_190742-93230b0d.pth'
        )
    }
    
    def __init__(self,
                 arch='tiny',
                 img_size=224,
                 patch_size=4,
                 in_channels=3,
                 window_size=7,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 out_indices=(0, 1, 2, 3),
                 out_after_downsample=False,
                 norm_eval=False,
                 pad_small_map=False):
        super().__init__()

        arch = arch.lower()
        self.arch = arch
        assert arch in set(self.arch_zoo), \
            f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
        self.arch_settings = self.arch_zoo[arch]

        self.embed_dims = self.arch_settings['embed_dims']
        self.depths = self.arch_settings['depths']
        self.num_heads = self.arch_settings['num_heads']
        self.num_layers = len(self.depths)
        self.out_indices = out_indices
        self.out_names = [f"stage{i+1}" for i in out_indices]
        self.out_after_downsample = out_after_downsample

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=self.embed_dims,
            kernel_size=patch_size,
            stride=patch_size,
            input_size=img_size
        )
        self.patch_resolution = self.patch_embed.init_out_size

        self._register_load_state_dict_pre_hook(
            self._prepare_relative_position_bias_table)

        self.drop_after_pos = nn.Dropout(p=drop_rate)
        self.norm_eval = norm_eval

        # stochastic depth
        total_depth = sum(self.depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]  # stochastic depth decay rule

        self.stages = nn.ModuleList()
        embed_dims = [self.embed_dims]
        for i, (depth,
                num_heads) in enumerate(zip(self.depths, self.num_heads)):

            downsample = True if i < self.num_layers - 1 else False
            stage = SwinBlockSequence(
                embed_dims=embed_dims[-1],
                depth=depth,
                num_heads=num_heads,
                window_size=window_size,
                downsample=downsample,
                pad_small_map=pad_small_map
            )
            self.stages.append(stage)

            dpr = dpr[depth:]
            embed_dims.append(stage.out_channels)

        if self.out_after_downsample:
            self.num_features = embed_dims[1:]
        else:
            self.num_features = embed_dims[:-1]

        for i in out_indices:
            norm_layer = nn.LayerNorm(self.num_features[i])
            self.add_module(f'norm{i}', norm_layer)

    def use_pretrained(self, log=True):
        state_dict = load_state_dict_from_url(url=self.url_setting[self.arch])
        state_dict_ = { name.replace('backbone.', '') : weight for name, weight in state_dict.items()}
        incompatible = self.load_state_dict(state_dict_, strict=False)
        if log:
            log_incompatible_keys(incompatible)
    
    def forward(self, x):
        x, hw_shape = self.patch_embed(x)

        x = self.drop_after_pos(x)

        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape = stage(
                x, hw_shape, do_downsample=self.out_after_downsample)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(x)
                out = out.view(-1, *hw_shape,
                               self.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                outs.append(out)
            if stage.downsample is not None and not self.out_after_downsample:
                x, hw_shape = stage.downsample(x, hw_shape)

        return {name : out for name, out in zip(self.out_names, outs)}

    def _prepare_relative_position_bias_table(self, state_dict, prefix, *args,
                                              **kwargs):
        state_dict_model = self.state_dict()
        all_keys = list(state_dict_model.keys())
        for key in all_keys:
            if 'relative_position_bias_table' in key:
                ckpt_key = prefix + key
                if ckpt_key not in state_dict:
                    continue
                relative_position_bias_table_pretrained = state_dict[ckpt_key]
                relative_position_bias_table_current = state_dict_model[key]
                L1, nH1 = relative_position_bias_table_pretrained.size()
                L2, nH2 = relative_position_bias_table_current.size()
                if L1 != L2:
                    src_size = int(L1**0.5)
                    dst_size = int(L2**0.5)
                    new_rel_pos_bias = resize_relative_position_bias_table(
                        src_size, dst_size,
                        relative_position_bias_table_pretrained, nH1)
            
                    logger = logging.getLogger(__name__)
                    logger.info('Resize the relative_position_bias_table from '
                                f'{state_dict[ckpt_key].shape} to '
                                f'{new_rel_pos_bias.shape}')
                    state_dict[ckpt_key] = new_rel_pos_bias

                    # The index buffer need to be re-generated.
                    index_buffer = ckpt_key.replace('bias_table', 'index')
                    del state_dict[index_buffer]


def resize_relative_position_bias_table(src_shape, dst_shape, table, num_head):
    """Resize relative position bias table.

    Args:
        src_shape (int): The resolution of downsampled origin training
            image, in format (H, W).
        dst_shape (int): The resolution of downsampled new training
            image, in format (H, W).
        table (tensor): The relative position bias of the pretrained model.
        num_head (int): Number of attention heads.

    Returns:
        torch.Tensor: The resized relative position bias table.
    """
    import numpy as np
    from scipy import interpolate

    def geometric_progression(a, r, n):
        return a * (1.0 - r**n) / (1.0 - r)

    left, right = 1.01, 1.5
    while right - left > 1e-6:
        q = (left + right) / 2.0
        gp = geometric_progression(1, q, src_shape // 2)
        if gp > dst_shape // 2:
            right = q
        else:
            left = q

    dis = []
    cur = 1
    for i in range(src_shape // 2):
        dis.append(cur)
        cur += q**(i + 1)

    r_ids = [-_ for _ in reversed(dis)]

    x = r_ids + [0] + dis
    y = r_ids + [0] + dis

    t = dst_shape // 2.0
    dx = np.arange(-t, t + 0.1, 1.0)
    dy = np.arange(-t, t + 0.1, 1.0)

    all_rel_pos_bias = []

    for i in range(num_head):
        z = table[:, i].view(src_shape, src_shape).float().numpy()
        f_cubic = interpolate.interp2d(x, y, z, kind='cubic')
        all_rel_pos_bias.append(
            torch.Tensor(f_cubic(dx,
                                 dy)).contiguous().view(-1,
                                                        1).to(table.device))
    new_rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
    return new_rel_pos_bias
