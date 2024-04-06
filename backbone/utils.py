import os
import sys
import requests
from tqdm import tqdm

import logging
import torch
import torch.nn as nn
from termcolor import colored
from collections import OrderedDict


def load_state_dict_from_url(url: str, model_dir=None):
    """
    NOTE:下载预训练权重文件并加载权重

    url: 权重文件下载链接.
    model_dir: 权重文件保存文件夹
    """
    if model_dir is None:
        hub_dir = os.getcwd()  # 当前工作路径
        model_dir = os.path.join(hub_dir, 'checkpoints')
        os.makedirs(model_dir, exist_ok=True)  # 创建checkpoints文件夹
    file_name = url.split('/')[-1]
    dst_file = os.path.join(model_dir, file_name)
    if not os.path.exists(dst_file):
        sys.stderr.write(f"Downloading: '{url}' \nto '{dst_file}'\n")
        # 用流stream的方式获取url的数据
        try:
            resp = requests.get(url, stream=True, timeout=10)  # 10秒超时
            # 拿到文件的长度，并把total初始化为0
            total = int(resp.headers.get('content-length', 0))
            # 初始化tqdm，传入总数，文件名等数据，接着就是写入，更新等操作了
            with open(dst_file, 'wb') as file, tqdm(
                total=total,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
                ncols=70,
            ) as bar:
                for data in resp.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)
        except requests.Timeout:
            print(f"请求超时！尝试手动下载权重文件:\n{url}\n到 '{model_dir}'")

    assert os.path.exists(dst_file), "预训练权重文件不存在！"
    load_dict = torch.load(dst_file)
    if isinstance(load_dict, OrderedDict):
        return load_dict  # 下载的checkpoint仅为权重文件
    elif 'state_dict' in load_dict.keys():
        return load_dict['state_dict']  # 下载的checkpoint包含其他信息，如model,meta等
    elif 'model' in load_dict.keys():
        return load_dict['model']


# 获取模型中间层输出
class IntermediateLayerGetter(nn.ModuleDict):
    """
    Args
    model: nn.module
    return_layers: dict, eg {"layer1":"res2", "layer2":"res3"}
    """

    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


# 记录模型与预训练权重不匹配的键
def log_incompatible_keys(incompatible):
    logger = logging.getLogger()
    missing_keys = incompatible.missing_keys
    unexpected_keys = incompatible.unexpected_keys
    if len(missing_keys) > 0:
        miss_msg = "Some model parameters or buffers are not found in the checkpoint:\n"
        miss_msg += "\n".join([colored(x, "cyan") for x in missing_keys])
        logger.warning(miss_msg)
    if len(unexpected_keys) > 0:
        notuse_msg = "The checkpoint state_dict contains keys that are not used by the model:\n"
        notuse_msg += "\n".join([colored(x, "magenta")
                                for x in unexpected_keys])
        logger.warning(notuse_msg)


# 统计模型所有待学习参数
def count_params(model: nn.Module):
    params = round(sum(p.numel()
                   for p in model.parameters() if p.requires_grad) / 1e6, 2)
    print(f'Total parameters: {params}M')
