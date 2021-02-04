from typing import Tuple
from torch import nn
import torch

class LSTMEncoder(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers: int = 1):
        super().__init__()

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, sequences) -> Tuple[torch.Tensor, torch.Tensor]:
        sequences, sent_len = nn.utils.rnn.pad_packed_sequence(sequences, batch_first=True)
        sequences = sequences.squeeze(2)
        sequences = self.word_embeddings(sequences)
        # embeds = torch.transpose(embeds, 0, 1)
        sequences = nn.utils.rnn.pack_padded_sequence(sequences, sent_len, batch_first=True, enforce_sorted=False)
        sequences, _ = self.lstm(sequences)
        return nn.utils.rnn.pad_packed_sequence(sequences, batch_first=True)
