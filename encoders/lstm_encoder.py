from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch import Tensor

from encoders.utils import last_timestep

class LSTMEncoder(nn.Module):

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.out_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, sequences) -> Tensor:
        sequences, sent_len = pad_packed_sequence(sequences, batch_first=True)
        sequences = sequences.squeeze(2)
        sequences = self.word_embeddings(sequences)

        sequences = pack_padded_sequence(sequences, sent_len, batch_first=True, enforce_sorted=False)
        sequences, _ = self.lstm(sequences)

        eos_embs = last_timestep(*pad_packed_sequence(sequences, batch_first=True))

        return eos_embs