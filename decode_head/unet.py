
# Implementation of: 'U-Net: Convolutional Networks for Biomedical Image Segmentation'
# https://arxiv.org/abs/1505.04597

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .losses import DiceLoss


class DoubleConv(nn.Module):
    """
    Double convolution Block 
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.in_channels = in_ch
        self.channels = out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()
        self.in_channels = in_ch
        self.channels = out_ch

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_ch, out_ch, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x
    


class UNetHead(nn.Module):
    def __init__(self,
                 num_classes: int = 4,
                 class_weights : List[int] =  [1, 1, 2, 4],
                 in_channels: List[int] = [256, 512, 1024, 2048],
                 common_stide: int = 4,
                 deep_supervision=True):
        super().__init__()
        self.common_stride = common_stide
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.deep_supervision = deep_supervision
        self.in_channels = in_channels

        self.num_stages = len(self.in_channels)
        for i in range(self.num_stages):
            if i == self.num_stages - 1: # 最后一个stage输入无concat
                conv_layers = DoubleConv(self.in_channels[i], self.in_channels[i])
            else:
                conv_layers = DoubleConv(self.in_channels[i] * 2, self.in_channels[i])
            self.add_module(f"deconv{i+1}", conv_layers)
            if i != 0: # 第一个stage输入无up
                up_layer = UpConv(self.in_channels[i], self.in_channels[i-1])
                self.add_module(f"up{i+1}", up_layer)

        if deep_supervision: # 深度监督,每个decode stage预测
            for i in range(self.num_stages):
                self.add_module(
                    f"pred{i+1}", nn.Conv2d(self.in_channels[i], num_classes, 1))
        else:
            self.add_module(f"pred{self.num_stages}", nn.Conv2d(
                self.in_channels[0], num_classes, 1))

    def forward(self, x, targets=None):
        """
        Returns:
            In training, returns (dict of losses)
            In inference, returns (CxHxW logits)
        """
        x = self._forward_layers(x)
        if targets is not None:
            return self.losses(x, targets)
        else:
            logits = F.interpolate(
                x[-1], scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            return logits

    def _forward_layers(self, input_features: List[torch.Tensor]):
        assert len(input_features) == self.num_stages
        outs = []
        for i in range(self.num_stages-1, -1, -1): # 倒序索引,逐步解码
            deconv_layers = getattr(self, f"deconv{i+1}")
            pred_layer = getattr(self, f"pred{i+1}", None)
            up_layer = getattr(self, f"up{i+1}", None)
            if i == self.num_stages - 1:
                x = deconv_layers(input_features[i])
            else:
                x = deconv_layers(torch.cat((input_features[i], x), dim=1))
            if pred_layer is not None:
                outs.append(pred_layer(x))
            if up_layer is not None:
                x = up_layer(x)
        return outs

    def losses(self, preds: List[torch.Tensor], target: torch.tensor):
        lambda_ = 0.5
        dice = DiceLoss(class_weight=self.class_weights)
        loss = 0.0
        for pred in preds:
            pred = F.interpolate(
                pred, size=target.shape[-2:], mode="bilinear", align_corners=True)
            loss += (1 - lambda_) * F.cross_entropy(pred, target,
                                                    weight=pred.new_tensor(self.class_weights)) +  \
                lambda_ * dice(pred, target)
        return {"loss_sem_seg": loss / len(preds)}
    