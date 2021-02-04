from torch import nn, Tensor

class DenseEncoder(nn.Module):

    def __init__(self, in_size, out_size):
        super().__init__()

        self.batchnorm = nn.BatchNorm1d(in_size)
        self.dense = nn.Linear(in_size, out_size)
        self.relu = nn.ReLU()

    def forward(self, embs: Tensor) -> Tensor:
        x = self.batchnorm(embs)
        x = self.dense(x)
        return self.relu(x)