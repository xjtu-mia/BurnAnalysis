# 'MaxViT: Multi-Axis Vision Transformer'
#       - https://arxiv.org/pdf/2204.01697.pdf

import os
try:
    import timm
except ImportError:
    os.system('pip install timm==0.9.10')

import math
from typing import List
from torch import nn
import torch.nn.functional as F

from timm.models.maxxvit import MaxxVitTransformerCfg
from timm.models.maxxvit import PartitionAttentionCl as PartitionAttentionCl_


class PartitionAttentionCl(PartitionAttentionCl_):
    def forward(self, x):
        B, H, W, C = x.size()
        if H % self.partition_size[0] > 0 or W % self.partition_size[1] > 0:
            new_h = math.ceil(H / self.partition_size[0]) * self.partition_size[0]
            new_w = math.ceil(W / self.partition_size[1]) * self.partition_size[1]
            x = x.permute(0, 3, 1, 2) # BCHW
            x = F.pad(x, [0, new_w - W, 0, new_h - H]) # [left, right, top, bottom]
            x = x.permute(0, 2, 3, 1) # BHWC
            # -----------------------------------------
            x = x + self.drop_path1(self.ls1(self._partition_attn(self.norm1(x))))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
            # -----------------------------------------
            x = x[:, :H, :W, :]
        else:
            x = x + self.drop_path1(self.ls1(self._partition_attn(self.norm1(x))))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class MaxVit(nn.Module):

    """
    Multi-Axis Vision Transformer, rewrapper of implementation in timm
    """

    arch_zoo = {
        **dict.fromkeys(['t', 'tiny'],
                        'maxvit_tiny_tf_384'),
        **dict.fromkeys(['s', 'small'],
                        'maxvit_small_tf_384'),
        **dict.fromkeys(['b', 'base'],
                        'maxvit_base_tf_384'),
    }  
        
    file_setting = {
        **dict.fromkeys(
            ['t', 'tiny'],
            {'file' : 'maxvit_tiny_tf_384_in1k.safetensors',
             'url' : 'https://huggingface.co/timm/maxvit_tiny_tf_384.in1k/tree/main'}
        ),
        **dict.fromkeys(
            ['s', 'small'],
            {'file' : 'maxvit_small_tf_384_in1k.safetensors',
             'url' : 'https://huggingface.co/timm/maxvit_small_tf_384.in1k/tree/main'}
        ),
        **dict.fromkeys(
            ['b', 'base'],
            {'file' : 'maxvit_base_tf_384_in1k.safetensors',
             'url' : 'https://huggingface.co/timm/maxvit_base_tf_384.in1k/tree/main'}
        )
    }

    def __init__(self,
                 arch: str = 'tiny',
                 out_indices: List[int] = [0, 1, 2, 3]):
        super().__init__()
        self.arch = arch.lower()
        maxvit = timm.create_model(self.arch_zoo[self.arch])
        self.stem = maxvit.stem
        stages = maxvit.stages
        replace_layer(stages)
        self.stages = stages
        self.norm = maxvit.norm

        self.out_indices = out_indices
        self.out_names = [f"stage{i+1}" for i in out_indices]

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for stage in self.stages:
            x = stage(x)
            outs.append(x)
        return {name : out for name, out in zip(self.out_names, outs)}

    def use_pretrained(self, log=True):
        try: import safetensors
        except ImportError:
            os.system('pip install safetensors==0.4.0')
        from safetensors.torch import load_file
        from .utils import log_incompatible_keys
        ckpt_file = f"checkpoints/{self.file_setting[self.arch].get('file')}"
        if os.path.exists(ckpt_file):
            state_dict = load_file(ckpt_file)
            incompatible = self.load_state_dict(state_dict, strict=False)
            if log:
                log_incompatible_keys(incompatible)
        else:
            print(f"预训练模型文件:{ckpt_file}不存在!\n下载链接:\n{self.file_setting[self.arch].get('url')}")


def replace_layer(model: nn.Module):
    # 创建一个临时的PartitionAttentionCl实例
    # 重写了其父类的前向传播函数（支持任意尺寸输入），用于替换模型中的相应层
    tmp_module = PartitionAttentionCl(dim=64, 
                                        cfg = MaxxVitTransformerCfg(window_size=[8,8]))
    for name, child in model.named_children():
        if isinstance(child, PartitionAttentionCl_):
            tmp_module.__dict__.update(child.__dict__) # 获取当前模块的参数,并更新至新的模型
            setattr(model, name, tmp_module)
        else:
            replace_layer(child)
