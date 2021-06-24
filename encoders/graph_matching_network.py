from typing import List

from torch import nn
from torch_geometric.data.batch import Batch
from torch_geometric.nn.glob.glob import global_mean_pool

from encoders.graph_matching_layer import GINMatchingLayer, GraphMatchingLayer


def _get_layer(layer_type: str):
    if layer_type == "gin":
        return GINMatchingLayer
    if layer_type == "gml":
        return GraphMatchingLayer
    raise ValueError


class GraphMatchingNetwork(nn.Module):

    def __init__(self, node_labels: int, input_dim: int, hidden_dims: List[int], layer_type: str = "gin", **layer_kwargs):
        super().__init__()
        self.node_embeddings = nn.Embedding(node_labels, input_dim)

        prev_dim = input_dim
        self.layers = nn.ModuleList()
        for hd in hidden_dims:
            self.layers.append(_get_layer(layer_type)(prev_dim, [hd, hd], **layer_kwargs))
            prev_dim = hd

        self.encoder = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], hidden_dims[-1])
        )


    def forward(self, x_data: Batch, y_data: Batch):

        x, x_edge_index, x_batch = x_data.x, x_data.edge_index, x_data.batch
        y, y_edge_index, y_batch = y_data.x, y_data.edge_index, y_data.batch

        x = self.node_embeddings(x).squeeze(1)
        y = self.node_embeddings(y).squeeze(1)

        for l in self.layers:
            x, y = l(x, x_edge_index, x_batch, y, y_edge_index, y_batch)

        x = global_mean_pool(x, x_batch)
        y = global_mean_pool(y, y_batch)

        return self.encoder(x), self.encoder(y)
