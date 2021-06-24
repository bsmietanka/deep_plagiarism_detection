from typing import Optional, Sequence, Union

import torch
from torch import Tensor
from torch import nn
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.data import Batch
from torch_geometric.nn import GINConv, global_mean_pool



class GIN0(nn.Module):
    def __init__(self, node_features: int, hidden_dim: int, num_layers: int = 5, train_eps: bool = False):
        super().__init__()
        self.conv1 = GINConv(
            Sequential(
                Linear(node_features, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                BN(hidden_dim),
            ), train_eps=train_eps)
        self.convs = nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden_dim, hidden_dim),
                        ReLU(),
                        Linear(hidden_dim, hidden_dim),
                        ReLU(),
                        BN(hidden_dim),
                    ), train_eps=train_eps))
        self.lins = nn.ModuleList()
        # for i in range(num_layers):
        for i in range(1):
            self.lins.append(nn.Sequential(Linear(hidden_dim, hidden_dim), nn.Tanh()))
        # self.lin1 = Linear(hidden_dim, hidden_dim)


    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        # self.lin1.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor):
        # embs = []
        x = self.conv1(x, edge_index)
        # embs.append(global_mean_pool(self.lins[0](x), batch))
        for i, conv in enumerate(self.convs):
            new_x = conv(x, edge_index)
            x = x + new_x
            # embs.append(global_mean_pool(self.lins[i + 1](x), batch))
        return global_mean_pool(self.lins[-1](x), batch)
        # return torch.cat(embs, dim=1)
        # return embs[-1]

    def __repr__(self):
        return self.__class__.__name__


class GraphEncoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 node_labels: int = -1,
                 num_layers: int = 5,
                 node_embeddings: Optional[int] = None,
                 train_eps: bool = False):
        super().__init__()
        # self.out_dim = hidden_dim

        if node_embeddings is not None:
            self.node_embeddings = nn.Embedding(node_labels, embedding_dim=node_embeddings)
        else:
            self.node_embeddings = nn.Identity()

        node_features = node_embeddings if node_embeddings is not None else input_dim
        self.model = GIN0(node_features, hidden_dim, num_layers, train_eps)
        self.out_dim = len(self.model.lins) * hidden_dim


    def forward(self, data: Batch) -> torch.FloatTensor:
        x = self.node_embeddings(data.x).float()
        if len(x.shape) == 3:
            x = x.squeeze()

        x = self.model(x, data.edge_index, data.batch)

        return x
