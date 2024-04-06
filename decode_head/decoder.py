import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple

from modules.dscu import DSCUBlcok
from modules.attention_gate import Attention_block
from .losses import DiceLoss


class ASPPModule(nn.ModuleList):
    """Atrous Spatial Pyramid Pooling (ASPP) Module.

    Args:
        dilations (tuple[int]): Dilation rate of each layer.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
    """

    def __init__(self, 
                 dilations: Tuple[int], in_channels: int, channels: int):
        super().__init__()
        self.dilations = dilations
        self.in_channels = in_channels
        self.channels = channels

        for dilation in dilations:
            self.append(
                nn.Sequential(
                    nn.Conv2d(self.in_channels, 
                              self.channels, 
                              kernel_size =1 if dilation == 1 else 3, 
                              dilation=dilation, 
                              padding=0 if dilation == 1 else dilation, 
                              bias=False),
                    nn.BatchNorm2d(self.channels),
                    nn.ReLU(inplace=True)
                )
            )

    def forward(self, x):
        outs = []
        for aspp_module in self:
            outs.append(aspp_module(x))
        return outs
    

class BurnDecodeHead(nn.Module):
    def __init__(self, num_classes: int,
                 in_channels: List[int] = [128, 256, 512, 1024],
                 dilations: List[int] = [1, 6, 12, 18],
                 strides: List[int] = [4, 8, 16, 32],
                 conv_dim: int = 128,
                 class_weights: List[int] = [1, 1, 2, 4],
                 fuse_type: str = 'add',
                 up_type: str = 'interpolation',
                 with_aspp: bool = True):
        """
        args:
        fuse_type|str: 'add' or 'concat'
        add_type|str: 'interpolation' or 'deconv' or 'dscu'
        """
        super().__init__()

        self.num_classes = num_classes
        self.strides = strides
        self.in_channels = in_channels
        self.dilations = dilations
        self.conv_dim = conv_dim
        self.class_weights = class_weights
        self.up_type = up_type
        self.fuse_type = fuse_type
        self.with_aspp = with_aspp

        if with_aspp:
            self.aspp = ASPPModule(dilations, in_channels=in_channels[-1], channels=in_channels[-1] // 4)
            self.image_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels[-1], in_channels[-1] // 4, 1),
                nn.BatchNorm2d(in_channels[-1] // 4),
                nn.ReLU(inplace=True)
                )
            
            self.aspp_proj = nn.Sequential(
                nn.Conv2d(
                    in_channels[-1] // 4 * (len(dilations) + 1),
                    conv_dim,
                    3,
                    padding=1
                ),
                nn.BatchNorm2d(conv_dim),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv4 = nn.Sequential(
                nn.Conv2d(
                    in_channels[-1], conv_dim, 3, padding=1),
                nn.BatchNorm2d(conv_dim),
                nn.ReLU(inplace=True)
            )
        if self.up_type == 'interpolation':
            self.upsample_4 = nn.Upsample(scale_factor=2, mode='bilinear')
            self.upsample_3 = nn.Upsample(scale_factor=2, mode='bilinear')
            self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear')
        elif self.up_type == 'deconv':
            self.upsample_4 = nn.ConvTranspose2d(conv_dim, conv_dim, kernel_size=2, stride=2)
            self.upsample_3 = nn.ConvTranspose2d(conv_dim, conv_dim, kernel_size=2, stride=2)
            self.upsample_2 = nn.ConvTranspose2d(conv_dim, conv_dim, kernel_size=2, stride=2)
        elif self.up_type == 'dscu':
            self.upsample_4 = DSCUBlcok(conv_dim, conv_dim)
            self.upsample_3 = DSCUBlcok(conv_dim, conv_dim)
            self.upsample_2 = DSCUBlcok(conv_dim, conv_dim)

        self.lateral_3  = nn.Conv2d(in_channels[-2], conv_dim, 1)
        self.lateral_2  = nn.Conv2d(in_channels[-3], conv_dim, 1)
        self.lateral_1  = nn.Conv2d(in_channels[-4], conv_dim, 1)

        if self.fuse_type == 'add':
            self.conv3 = nn.Sequential(
                nn.Conv2d(conv_dim, conv_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(conv_dim),
                nn.ReLU(inplace=True)
            )

            self.conv2 = nn.Sequential(
                nn.Conv2d(conv_dim, conv_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(conv_dim),
                nn.ReLU(inplace=True)
            )

            self.conv1 = nn.Sequential(
                nn.Conv2d(conv_dim, conv_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(conv_dim),
                nn.ReLU(inplace=True)
            )

        elif self.fuse_type == 'concat':
            self.conv3 = nn.Sequential(
                nn.Conv2d(conv_dim * 2, conv_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(conv_dim),
                nn.ReLU(inplace=True)
            )

            self.conv2 = nn.Sequential(
                nn.Conv2d(conv_dim * 2, conv_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(conv_dim),
                nn.ReLU(inplace=True)
            )

            self.conv1 = nn.Sequential(
                nn.Conv2d(conv_dim * 2, conv_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(conv_dim),
                nn.ReLU(inplace=True)
            )

        self.proj = nn.Sequential(
                nn.Conv2d(conv_dim, conv_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(conv_dim),
                nn.ReLU(inplace=True)
            )
        
        self.outc = nn.Conv2d(conv_dim, num_classes, 1)

    
    def forward(self, features: List[torch.Tensor], targets = None):
        x = self._forward_layers(features)
        if targets is not None:
            return self.losses(x, targets)
        else:
            logits = F.interpolate(
                x, 
                scale_factor=self.strides[0], 
                mode="bilinear", align_corners=False
            )
            return logits
    
    def _forward_layers(self, features: List[torch.Tensor]):
        x1, x2, x3, x4 = features
        if self.with_aspp:
            aspp_outs = self.aspp(x4)

            aspp_outs.append(F.interpolate(self.image_pool(x4), size=x4.size()[-2:],
                                        mode='bilinear', align_corners=False))
            
            x4 = self.aspp_proj(torch.cat(aspp_outs, dim=1))
        else:
            x4 = self.conv4(x4)
        x3 = self.lateral_3(x3)
        x2 = self.lateral_2(x2)
        x1 = self.lateral_1(x1)

        x4 = self.upsample_4(x4)
        if self.fuse_type == 'add':
            x3 = self.conv3(x4 + x3)
        elif self.fuse_type == 'concat':
            x3 = self.conv3(torch.cat([x4, x3], dim=1))

        x3 = self.upsample_3(x3)
        if self.fuse_type == 'add':
            x2 = self.conv2(x3 + x2)
        elif self.fuse_type == 'concat':
            x2 = self.conv2(torch.cat([x3, x2], dim=1)) 

        x2 = self.upsample_2(x2)
        if self.fuse_type == 'add':
            x1 = self.conv1(x2 + x1)
        elif self.fuse_type == 'concat':
            x1 = self.conv1(torch.cat([x2, x1], dim=1))  

        x4 = F.interpolate(x4, x1.size()[-2:], mode='bilinear')
        x3 = F.interpolate(x3, x1.size()[-2:], mode='bilinear')
        x2 = F.interpolate(x2, x1.size()[-2:], mode='bilinear')

        # proj_out = self.proj(torch.cat([x1, x2, x3, x4], dim=1))
        proj_out = self.proj(x1 + x2 + x3 + x4)
        out = self.outc(proj_out)        
        return out
    
    def losses(self, preds: torch.Tensor, targets: torch.tensor):
        lambda_ = 0.5
        dice = DiceLoss(class_weight=self.class_weights)
        preds = F.interpolate(
                    preds, size=targets.shape[-2:], mode="bilinear", align_corners=True)
        loss = (1 - lambda_) * F.cross_entropy(preds, targets, 
                                               weight=preds.new_tensor(self.class_weights)) +  \
                        lambda_ * dice(preds, targets)
        return {"loss_sem_seg": loss}


class BurnDecodeAGHead(BurnDecodeHead):
    def __init__(self, 
                guide_in_channels: List[int] = [256, 256, 256, 256], 
                **kwargs):
        super().__init__(**kwargs)

        for i in range(len(guide_in_channels)):
            self.add_module(f"attention_gate_{i + 1}",
                            Attention_block(guide_in_channels[i],
                                            self.conv_dim,
                                            self.conv_dim)
                            )
            

    def _forward_layers(self, features: List[torch.Tensor]):
        x1, x2, x3, x4, p1, p2, p3, p4 = features
        if self.with_aspp:
            aspp_outs = self.aspp(x4)

            aspp_outs.append(F.interpolate(self.image_pool(x4), size=x4.size()[-2:],
                                        mode='bilinear', align_corners=False))
            
            x4 = self.aspp_proj(torch.cat(aspp_outs, dim=1))
        else:
            x4 = self.conv4(x4)
        x3 = self.lateral_3(x3)
        x2 = self.lateral_2(x2)
        x1 = self.lateral_1(x1)

        x4 = self.attention_gate_4(p4, x4)
        x3 = self.attention_gate_3(p3, x3)
        x2 = self.attention_gate_2(p2, x2)
        x1 = self.attention_gate_1(p1, x1)

        x4 = self.upsample_4(x4)
        if self.fuse_type == 'add':
            x3 = self.conv3(x4 + x3)
        elif self.fuse_type == 'concat':
            x3 = self.conv3(torch.cat([x4, x3], dim=1))

        x3 = self.upsample_3(x3)
        if self.fuse_type == 'add':
            x2 = self.conv2(x3 + x2)
        elif self.fuse_type == 'concat':
            x2 = self.conv2(torch.cat([x3, x2], dim=1)) 

        x2 = self.upsample_2(x2)
        if self.fuse_type == 'add':
            x1 = self.conv1(x2 + x1)
        elif self.fuse_type == 'concat':
            x1 = self.conv1(torch.cat([x2, x1], dim=1))  

        x4 = F.interpolate(x4, x1.size()[-2:], mode='bilinear')
        x3 = F.interpolate(x3, x1.size()[-2:], mode='bilinear')
        x2 = F.interpolate(x2, x1.size()[-2:], mode='bilinear')

        # proj_out = self.proj(torch.cat([x1, x2, x3, x4], dim=1))
        proj_out = self.proj(x1 + x2 + x3 + x4)
        out = self.outc(proj_out)        
        return out