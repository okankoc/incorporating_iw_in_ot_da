import torch
from torch import nn

# 3rd party library for multi class hinge loss
from dfw.losses import MultiClassHingeLoss

class MarginLoss(nn.Module):
    def __init__(self):
        super(MarginLoss, self).__init__()
        self.loss = MultiClassHingeLoss()
        self.reduction = 'mean'

    # Assumes it gets one-hot-labels (all other loss functions do)
    def forward(self, x, y):
        losses = self.loss(x, torch.argmax(y, dim=1))
        if self.reduction == 'mean':
            return torch.mean(losses)
        return losses


class EuclideanLoss(nn.Module):
    def __init__(self):
        super(EuclideanLoss, self).__init__()
        self.reduction = 'mean'

    def forward(self, x, y):
        losses = torch.sqrt(torch.sum((x - y) ** 2, dim=1))
        if self.reduction == 'mean':
            return torch.mean(losses)
        return losses
