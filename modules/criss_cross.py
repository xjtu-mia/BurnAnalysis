# Implementation of: "CCNet: Criss-Cross Attention for Semantic Segmentation"
# https://arxiv.org/abs/1911.00359.pdf

import torch
import torch.nn as nn
import torch.nn.functional as F


def NEG_INF_DIAG(n: int, device: torch.device) -> torch.Tensor:
    """Returns a diagonal matrix of size [n, n].

    The diagonal are all "-inf". This is for avoiding calculating the
    overlapped element in the Criss-Cross twice.
    """
    return torch.diag(torch.tensor(float('-inf')).to(device).repeat(n), 0)


class CCBlock(nn.Module):
    """Criss-Cross Attention Module.
    Args:
        in_channels (int): Channels of the input feature maps.
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 8, gamma: float = 0, with_residual: bool = True):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1)
        # self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float)) 
        self.W_z = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1)
        self.in_channels = in_channels
        self.with_residual = with_residual


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward function of Criss-Cross Attention.

        Args:
            x (torch.Tensor): Input feature with the shape of
                (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output of the layer, with the shape of
            (batch_size, in_channels, height, width)
        """
        B, C, H, W = x.size()
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)
        energy_H = torch.einsum('bchw,bciw->bwhi', query, key) + NEG_INF_DIAG(
            H, query.device)
        energy_H = energy_H.transpose(1, 2)
        energy_W = torch.einsum('bchw,bchj->bhwj', query, key)
        attn = F.softmax(
            torch.cat([energy_H, energy_W], dim=-1), dim=-1)  # [B,H,W,(H+W)]
        out = torch.einsum('bciw,bhwi->bchw', value, attn[..., :H])
        out += torch.einsum('bchj,bhwj->bchw', value, attn[..., H:])
        out = self.W_z(out)
        # if self.with_residual:
        #     out = self.gamma * out + x
        # else:
        #     out = self.gamma * out
        if self.with_residual:
            out = out + x
        out = out.contiguous()
        return out
    