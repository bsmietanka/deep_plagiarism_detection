from typing import Optional, Sequence, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.normalization import LayerNorm
from torch_geometric import nn as pyg_nn
from torch_geometric.data import Batch


class GraphEncoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dims: Sequence[int],
                 num_node_embeddings: int = 4,
                 layer_type: str = "gcn",
                 post_mp_dim: Optional[int] = None):
        super().__init__()

        self.layer_type = layer_type.lower()

        self.node_embeddings = nn.Embedding(input_dim, embedding_dim=num_node_embeddings)

        self.dropout_rate = 0.25
        self.num_layers = len(hidden_dims)

        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        prev_dim = num_node_embeddings
        for i, hidden_dim in enumerate(hidden_dims, 1):
            self.convs.append(self._gnn_layer(prev_dim, hidden_dim))
            if i != self.num_layers:
                self.lns.append(LayerNorm(hidden_dim))
            prev_dim = hidden_dim

        # post-message-passing
        self.post_mp = self._post_mp(prev_dim, post_mp_dim)


    def _gnn_layer(self, input_dim: int, hidden_dim: int) -> pyg_nn.MessagePassing:
        if self.layer_type == "gcn":
            return pyg_nn.GCNConv(input_dim, hidden_dim)

        if self.layer_type == "conv":
            return pyg_nn.GraphConv(input_dim, hidden_dim)

        if self.layer_type == "gin":
            return pyg_nn.GINConv(
                nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim)
                    )
                )

        # Try TransformerConv, SAGEConv, GATConv?
        raise ValueError("Not supported GNN layer type")


    def _post_mp(self, input_dim: int, output_dim: Optional[int]) -> nn.Module:
        if output_dim is None:
            return nn.Sequential() # identity

        return nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.Dropout(self.dropout_rate),
                nn.Linear(input_dim, output_dim)
            )


    def forward(self, data: Batch) -> torch.FloatTensor:
        node, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.node_embeddings(node).squeeze()

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            if i < self.num_layers - 1:
                x = self.lns[i](x)

        x = pyg_nn.global_mean_pool(x, batch)

        x = self.post_mp(x)

        return x
