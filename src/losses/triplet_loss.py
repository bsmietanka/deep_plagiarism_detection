from torch import nn
import torch

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def _calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self._calc_euclidean(anchor, positive)
        distance_negative = self._calc_euclidean(anchor, negative)

        # TODO: should it be really ReLU?
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()
