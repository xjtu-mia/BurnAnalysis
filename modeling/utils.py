import torch.nn as nn
from typing import List
from decode_head.unet import UpConv
from modules.dscu import DSCUBlcok
from modules.coswin import CoSwin


# add CoSwin block to encoder network
def add_coswin(model: nn.Module, in_channles: List[int], 
               layers: int = 1,
               flags: List[bool] = [True, True, True, True],
                window_size: int= 7, att_type: str = 'NonLocal'):
    """
    NOTE: adding CoSwin blcok to the last block of each stage,
    'att_type' shoule be one of ['NonLocal', 'GlobalContext', 'CrissCross'].
    """
    stages = model.backbone.stages
    new_stages = nn.ModuleList()
    for stage, in_channel, flag in zip (stages, in_channles, flags):
        if flag:
            stage = list(stage.children())
            # 在最后一个block之前添加
            # stage.insert(-1, CoSwin(in_channel, window_size, att_type))
            for _ in range(layers):
                stage.append(CoSwin(in_channel, window_size, att_type))
            new_stages.append(nn.Sequential(*stage))
        else:
            new_stages.append(stage)
    model.backbone.stages = new_stages
    return model


def UpConvToDSCUBlock(model: nn.Module):
    """Replace UpConv block to DSCU block"""
    for name, child in model.named_children():
        if isinstance(child, UpConv):
            in_channels = child.in_channels
            channels = child.channels
            setattr(model, name, DSCUBlcok(in_channels, channels))
        else:
            UpConvToDSCUBlock(child)
