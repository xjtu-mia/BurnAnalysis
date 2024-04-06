import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1, class_weight=None, ignore_index=255, loss_weight=1.0, reduction='mean'):
        super().__init__()
        self.smooth = smooth
        self.class_weight=class_weight
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight
        self.reduction = reduction
    
    def forward(self, pred : torch.Tensor, target : torch.Tensor):
        """
        pred: torch.Tensor  N, C, H, W
        target: torch.Tensor N, H, W
        """
        assert pred.shape[0] == target.shape[0]   
        valid_mask = target != self.ignore_index # NHW
        valid_mask = valid_mask[..., None] # NHW1
        num_classes = pred.shape[1]
        target = F.one_hot(target, num_classes) # one-hot, NHWC 
        pred = F.softmax(pred, dim=1).permute(0, 2, 3, 1) # softmax, 并调整通道顺序 NHWC
        
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight) # new_tensor创建张量的device与src tenor相同
        else:
            class_weight = pred.new_ones(num_classes)
        inter = (pred * target * valid_mask * class_weight).flatten(1) # NHWC * NHWC * NHW1 * C -> NHxWxC
        total = ((pred + target) * class_weight).flatten(1)
        v = 2 * inter.sum(1)
        loss = 1 - (2 * inter.sum(1) + self.smooth) / (total.sum(1) + self.smooth)  # 平滑参数防止分母为0
        if self.reduction == 'mean':
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
