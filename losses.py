import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    """Weighted sum of soft Dice loss and binary cross-entropy, computed from logits.
    Dice counters the strong foreground/background imbalance; BCE gives stable
    voxel-wise gradients early in training."""

    def __init__(self, dice_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.w, self.smooth = dice_weight, smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        p = torch.sigmoid(logits).flatten(1)
        t = target.flatten(1)
        inter = (p * t).sum(1)
        dice = (2 * inter + self.smooth) / (p.sum(1) + t.sum(1) + self.smooth)
        return self.w * (1 - dice.mean()) + (1 - self.w) * self.bce(logits, target)
