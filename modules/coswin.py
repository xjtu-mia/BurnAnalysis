import torch
import math
from typing import Tuple
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from .global_context import GCBlock
from .non_local import NLBlock
from .criss_cross import CCBlock
from .transformer import TransformerBlock
from .layernorm import LayerNorm2d


def patchify(x: Tensor, window_size: int=7):
    """
    args: x input tensor(N,C,H,W)
    return: tensor(N*num_patch_h*num_path_w,C,patch_size,patch_size)
    """
    # patches = x.unfold(2, size=size, step=size) # (N,C,H/size,W,h_size)
    # patches = x.unfold(3, size=size, step=size) # (N,C,H/size,W/size,h_size,w_size)
    N, C, H, W = x.shape
    # (N,C,H/size,W/size,h_size,w_size)
    patches = x.unfold(2, size=window_size, step=window_size).unfold(3, size=window_size, step=window_size) 
    # (N,C,num_patch_h*num_patch_w,size,size)
    patches = patches.contiguous().view(N, C, -1, window_size, window_size) 
    # (N,num_patch_h*num_patch_w,C,size,size)
    patches = patches.permute(0, 2, 1, 3, 4) 
    # (N*num_patch_h*num_patch_w,C,size,size)
    # (i,j) patch of sample k <- patches[k*num_patch_h*num_patch_w + i*num_patch_w + j,C,size,size]
    patches = patches.contiguous().view(-1,C,window_size,window_size) 
    return patches


def unpatchify(x: Tensor, batch: int, ori_size: Tuple[int, int]):
    """
    args: tensor(N*num_patch_h*num_path_w,C,patch_size,patch_size)
    return: x input tensor(N,C,H,W)
    """
    _, C, patch_size, patch_size = x.shape
    H, W = ori_size
    num_patch_h = H // patch_size
    num_patch_w = W // patch_size
    # (N,C,num_patch_h*num_patch_w,size,size)
    x = x.view(batch, -1, C, patch_size, patch_size).permute(0,2,1,3,4) 
    # (N,C,num_patch_h,num_patch_w,size,size)
    x = x.contiguous().view(batch, C, num_patch_h, num_patch_w, patch_size, patch_size) 
    # (N,C,num_patch_h,size,num_patch_w,size)
    x = x.permute(0,1,2,4,3,5) 
    #(N,C,H,W)
    x = x.contiguous().view(batch, C, H, W) 
    return x


class CoSwin(nn.Module):
    """
    NOTE: 'att_type' shoule be one of ['NonLocal', 'GlobalContext', 'CrissCross']
    """

    def __init__(self, in_channels: int, window_size: int = 7, 
                 att_type: str = 'NonLocal'):
        super().__init__()
        self.window_size = window_size
        gamma = torch.tensor(0, dtype=float, requires_grad=True)
        self.gamma = nn.Parameter(gamma)
        self.layer_norm = LayerNorm2d(in_channels)
        if att_type == 'NonLocal':
            self.att = NLBlock(in_channels=in_channels,
                               inter_channels=in_channels // 4,
                               with_residual=False)
        elif att_type == 'GlobalContext':
            self.att = GCBlock(in_channels=in_channels,
                                    reduction_ratio=4, 
                                    fusion_types=('channel_mul',))
        elif att_type == 'CrissCross':
            self.att = CCBlock(in_channels=in_channels, 
                                    reduction_ratio=4,
                                    with_residual=False)
        elif att_type == 'Transformer':
            self.att = TransformerBlock(input_size=window_size**2)

    def forward(self, x):
        B, _, H, W = x.shape
        # side should be divided by the window_size
        h_ = math.ceil(H / self.window_size) * self.window_size
        w_ = math.ceil(W / self.window_size) * self.window_size
        x_ = F.adaptive_avg_pool2d(x, (h_, w_)) 
        # divided into patches, claculate self-attttion in each pathch
        # [N*W/w_size*H/w_size],C,w_size,w_size
        att_out1 = self.att(patchify(x_, window_size=self.window_size))
        # reorganization of patches (N,C,H,W)
        att_out1 = unpatchify(att_out1, batch=B, ori_size=x_.shape[-2:])
        # reflection padding, padding_size = window_size, shifted window
        left, right = self.window_size // 2, self.window_size - self.window_size // 2
        top, bottom = self.window_size // 2, self.window_size - self.window_size // 2
        x_ = F.pad(x_, pad=[left, right, top, bottom], mode='reflect')
        #  [N*(W/w_size+1)*(H/w_size+1)],C,w_size,w_size
        att_out2 = self.att(patchify(x_, window_size=self.window_size))
        # (N,C,H,W)
        att_out2 = unpatchify(att_out2, batch=B, ori_size=x_.shape[-2:]) 
        # crop
        att_out2 = att_out2[..., left : left + h_, top : top + w_]
        # att_out = torch.sigmoid(att_out1 * att_out2)
        out = att_out1 * att_out2
        # rescale to original size
        out = F.adaptive_avg_pool2d(out, x.shape[-2:])
        return  self.layer_norm(out) * self.gamma + x