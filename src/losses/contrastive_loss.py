from torch import nn
import torch

class ContrastiveLoss(nn.Module):

    def __init__(self, margin=1.0):
        self.margin = margin

    def _calc_euclidean(self, x1: torch.Tensor, x2: torch.Tensor):
        return (x1 - x2).pow(2).sum(1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, labels: torch.Tensor):
        euclidean_distance = self._calc_euclidean(x1, x2)
        return torch.mean((1 - labels) * torch.pow(euclidean_distance, 2) + 
            (labels) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
