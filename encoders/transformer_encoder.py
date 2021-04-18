import math

import torch
from torch import nn
from torch import Tensor
from torch.nn.utils.rnn import pad_packed_sequence

from encoders.utils import last_timestep

# NOTE: transformer encoder layer works only for batch second input
class TransformerEncoder(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers: int = 1, nhead=4):
        super().__init__()
        self.out_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim=embedding_dim, padding_idx=0)

        self.emb_dim = embedding_dim
        if self.emb_dim % 2 != 0:
            raise ValueError(f"Cannot use sin/cos positional encoding with "
                            f"odd dim (got dim={self.emb_dim:d})")
        enc_layer = nn.TransformerEncoderLayer(embedding_dim, nhead=nhead, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)


    # NOTE: taken from https://github.com/wzlxjtu/PositionalEncoding2D
    def positional_encoding1d(self, length):
        """
        :param length: length of positions
        :return: length*d_model position matrix
        """
        pe = torch.zeros(length, self.emb_dim)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, self.emb_dim, 2, dtype=torch.float) *
                            (-math.log(10000.0) / self.emb_dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe


    def forward(self, sequences) -> Tensor:
        sequences, sent_lens = pad_packed_sequence(sequences, batch_first=True)
        sequences = sequences.squeeze(2)
        sequences = self.word_embeddings(sequences)
        for i in range(sequences.shape[0]):
            sequences[i, :sent_lens[i]] += self.positional_encoding1d(sent_lens[i]).to(sequences.device)
        sequences = sequences.transpose(0, 1)
        sequences = self.transformer_encoder(sequences)

        # retrieve <EOS> embeddings
        return last_timestep(sequences.transpose(0, 1), sent_lens)
