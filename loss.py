import torch
from torch import nn

# 3rd party library for multi class hinge loss
from dfw.losses import MultiClassHingeLoss

class MarginLoss(nn.Module):
    def __init__(self):
        super(MarginLoss, self).__init__()
        self.loss = MultiClassHingeLoss()

    # Assumes it gets one-hot-labels (all other loss functions do)
    def forward(self, x, y, reduction='mean'):
        return self.loss(x, torch.argmax(y, dim=1), reduction)


class EuclideanLoss(nn.Module):
    def __init__(self):
        super(EuclideanLoss, self).__init__()

    def forward(self, x, y, reduction='mean'):
        losses = torch.sqrt(torch.sum((x - y) ** 2, dim=1))
        if reduction == 'mean':
            return torch.mean(losses)
        return losses



class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, y, reduction='mean'):
        losses = self.loss(x, y)
        if reduction == 'mean':
            return torch.mean(losses)
        return losses
