from typing import Literal

from models.utils import last_timestep
from models.encoders.lstm_encoder import LSTMEncoder
from models.encoders.transformer_encoder import TransformerEncoder
import torch
from torch import nn


class EuclideanSiameseClassificator(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 num_layers: int = 1,
                 num_classes: int = 1,
                 nhead: int = 4,
                 encoder_type: Literal['lstm', 'transformer'] = 'lstm'):
        super().__init__()

        if encoder_type == "lstm":
            self.encoder = LSTMEncoder(vocab_size, embedding_dim, hidden_dim, num_layers)
        else:
            self.encoder = TransformerEncoder(vocab_size, embedding_dim, hidden_dim, num_layers, nhead)

        self.batchnorm = nn.BatchNorm1d(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, fun1_sentences, fun2_sentences):
        enc1 = last_timestep(*self.encoder(fun1_sentences))
        enc2 = last_timestep(*self.encoder(fun2_sentences))

        # euclidean dist (without square root)
        enc_diff = (enc1 - enc2).pow(2)

        dist = self.batchnorm(enc_diff)
        dist = self.head(dist)

        return torch.sigmoid(dist.flatten())
