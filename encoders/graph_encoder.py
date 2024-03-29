import torch
from torch import Tensor
from torch import nn
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN, functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GINConv, global_mean_pool, GATv2Conv

def make_layer(in_features, out_features, type):
    if type == "gat":
        return GATv2Conv(in_features, out_features)
    else:
        return GINConv(
            Sequential(
                Linear(in_features, out_features),
                ReLU(),
                Linear(out_features, out_features),
                ReLU(),
                BN(out_features),
            ), train_eps=True
        )

class GIN0(nn.Module):
    def __init__(self, node_features: int, hidden_dim: int, num_layers: int = 5, layer_type: str = "gin", residual: bool = True):
        super().__init__()
        self.conv1 = make_layer(node_features, hidden_dim, layer_type)

        self.convs = nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(make_layer(hidden_dim, hidden_dim, layer_type))
        self.relu = nn.ReLU()
        self.residual = residual

        self.lin = nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            nn.Tanh()
            )


    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin.reset_parameters()


    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor):
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            new_x = self.relu(conv(x, edge_index))
            if self.residual:
                x = x + new_x
            else:
                x = new_x
        return global_mean_pool(self.lin(x), batch)


class GraphEncoder(nn.Module):
    def __init__(self,
                 num_tokens: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 num_layers: int = 5,
                 layer_type: str = "gin",
                 residual: bool = True):
        super().__init__()

        self.node_embeddings = nn.Embedding(num_tokens, embedding_dim=embedding_dim)
        self.model = GIN0(embedding_dim, hidden_dim, num_layers, layer_type, residual)
        self.out_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Hardtanh(),
        )


    def forward(self, data: Batch) -> torch.FloatTensor:
        x = self.node_embeddings(data.x).float()
        if len(x.shape) == 3:
            x = x.squeeze()

        x = self.model(x, data.edge_index, data.batch)

        return self.mlp(x)
