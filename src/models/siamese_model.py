import torch
from torch import nn

from models.partial.lstm_encoder import LSTMEncoder
from models.partial.dense_encoder import DenseEncoder

class SiameseModel(nn.Module):

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 target_size: int, num_layers: int = 1, num_classes: int = 1):
        super().__init__()

        self.lstm_encoder = LSTMEncoder(vocab_size, embedding_dim, hidden_dim, num_layers)
        self.dense_encoder = DenseEncoder(2 * hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm = nn.BatchNorm1d(target_size)
        self.head = nn.Linear(target_size, num_classes)

    def forward(self, fun1_sentences, fun2_sentences):
        enc1, seg_lens1 = self.encoder(fun1_sentences)
        enc2, seq_lens2 = self.encoder(fun2_sentences)

        emb1 = torch.cat([enc1, enc2], dim=1)
        emb2 = torch.cat([enc2, enc1], dim=1)

        

        distance = enc1 - enc2
        distance = self.dropout(distance)
        distance = self.batchnorm(distance)
        return torch.sigmoid(self.head(distance).flatten())
