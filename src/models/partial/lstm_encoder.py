from torch import nn
import torch.nn.functional as F
import torch

class BaseModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers: int = 1):
        super().__init__()

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, sentences):
        sentences, sent_len = nn.utils.rnn.pad_packed_sequence(sentences, batch_first=True)
        sentences: torch.Tensor = sentences.squeeze(2)
        embeds = self.word_embeddings(sentences)
        # embeds = torch.transpose(embeds, 0, 1)
        lstm_in = nn.utils.rnn.pack_padded_sequence(embeds, sent_len, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(lstm_in)
        lstm_out, seq_lens = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        return lstm_out, seq_lens
