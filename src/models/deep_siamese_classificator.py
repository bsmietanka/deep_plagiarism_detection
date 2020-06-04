import torch
from torch import nn

from models import SiameseLSTMEncoder
from models.partial.dense_encoder import DenseEncoder

class DeepSiameseClassificator(nn.Module):

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_layers: int = 1, num_classes: int = 1):
        super().__init__()

        self.encoder = SiameseLSTMEncoder(vocab_size, embedding_dim, hidden_dim, num_layers)
        self.dense_encoder = DenseEncoder(2 * hidden_dim, hidden_dim)

        self.batchnorm1 = nn.BatchNorm1d(hidden_dim)
        self.dense1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu1 = nn.ReLU()

        self.batchnorm2 = nn.BatchNorm1d(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, fun1_sentences, fun2_sentences):
        enc1 = self.encoder(fun1_sentences)
        enc2 = self.encoder(fun2_sentences)

        # concat two ways so comparing is symmetrical
        emb1 = torch.cat([enc1, enc2], dim=1)
        emb2 = torch.cat([enc2, enc1], dim=1)

        emb1 = self.dense_encoder(emb1)
        emb2 = self.dense_encoder(emb2)

        # sum because it's symmetrical unlike subtraction
        dist = enc1 + enc2

        dist = self.batchnorm1(dist)
        dist = self.dense1(dist)
        dist = self.relu1(dist)

        dist = self.batchnorm2(dist)
        dist = self.head(dist)

        return torch.sigmoid(dist.flatten())
