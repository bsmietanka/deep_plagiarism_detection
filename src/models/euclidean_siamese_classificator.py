import torch
from torch import nn

from models import SiameseLSTMEncoder

class EuclideanSiameseClassificator(nn.Module):

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_layers: int = 1, num_classes: int = 1):
        super().__init__()

        self.encoder = SiameseLSTMEncoder(vocab_size, embedding_dim, hidden_dim, num_layers)

        self.batchnorm = nn.BatchNorm1d(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def _diff(self, x1: torch.Tensor, x2: torch.Tensor):
        return (x1 - x2).pow(2)

    def forward(self, fun1_sentences, fun2_sentences):
        enc1 = self.encoder(fun1_sentences)
        enc2 = self.encoder(fun2_sentences)

        enc_diff = self._diff(enc1, enc2)

        dist = self.batchnorm(enc_diff)
        dist = self.head(dist)

        return torch.sigmoid(dist.flatten())
