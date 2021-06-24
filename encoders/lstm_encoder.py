from torch import nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence
from torch import Tensor

from encoders.utils import last_timestep, sequences_mean

class LSTMEncoder(nn.Module):

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.out_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.lstm_layers = nn.ModuleList([nn.LSTM(input_size=hidden_dim,
                                             hidden_size=hidden_dim,
                                             batch_first=True) for _ in range(num_layers - 1)])
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Hardtanh(),
        )

    def forward(self, sequences) -> Tensor:
        sequences, sent_len = pad_packed_sequence(sequences, batch_first=True)
        sequences = sequences.squeeze(2)
        sequences = self.word_embeddings(sequences)
        # sequences = self.dropout(sequences)
        sequences = pack_padded_sequence(sequences, sent_len, batch_first=True, enforce_sorted=False)
        sequences, (ht, _) = self.lstm1(sequences)

        for lstm_layer in self.lstm_layers:
            new_sequences, (ht, _) = lstm_layer(sequences)
            # seq1, len1 = pad_packed_sequence(sequences, batch_first=True)
            # seq2, len2 = pad_packed_sequence(new_sequences, batch_first=True)
            # seq1 += seq2
            # sequences = pack_padded_sequence(seq1, len1, batch_first=True, enforce_sorted=False)
            sequences = PackedSequence(sequences.data + new_sequences.data,
                                       sequences.batch_sizes,
                                       sequences.sorted_indices,
                                       sequences.unsorted_indices)

        # seq_embs = ht[-1]
        seq_embs = sequences_mean(*pad_packed_sequence(sequences, batch_first=True))
        seq_embs = self.mlp(seq_embs)

        # print(seq_embs)

        return seq_embs