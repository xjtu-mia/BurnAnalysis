import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict


class DynamicTaskWeightAlign(nn.Module):
    def __init__(self, num_tasks=2):
        super().__init__()
        sigmas = torch.ones(num_tasks, requires_grad=True)
        self.sigmas = torch.nn.Parameter(sigmas)

    def forward(self, losses: Dict[str, Tensor]):
        losses = {k: v * 0.5 / self.sigmas[0] ** 2 if "sem_seg" in k else v *
                     0.5 / self.sigmas[1] ** 2 for k, v in losses.items()}
        return losses, self.sigmas