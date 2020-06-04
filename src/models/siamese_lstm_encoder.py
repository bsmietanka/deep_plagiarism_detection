import torch
from torch import nn

from models.partial.lstm_encoder import LSTMEncoder


# TODO: check GRU
class SiameseLSTMEncoder(nn.Module):

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.lstm_encoder = LSTMEncoder(vocab_size, embedding_dim, hidden_dim, num_layers)

    def last_timestep(self, unpacked: torch.Tensor, lengths: torch.Tensor):
        # Index of the last output for each sequence.
        lengths = lengths.to(unpacked.device)
        idx = (lengths - 1).view(-1, 1).expand(unpacked.size(0),
                                               unpacked.size(2)).unsqueeze(1)
        return unpacked.gather(1, idx).squeeze()

    def forward(self, fun1_sentences, fun2_sentences):
        enc1, seq_lens1 = self.lstm_encoder(fun1_sentences)
        enc2, seq_lens2 = self.lstm_encoder(fun2_sentences)

        enc1 = self.last_timestep(enc1, seq_lens1)
        enc2 = self.last_timestep(enc2, seq_lens2)

        return enc1, enc2
