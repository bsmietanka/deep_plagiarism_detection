from typing import List, Tuple
import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gin_conv import GINConv
from torch_geometric.typing import Size



def pairwise_euclidean_similarity(x, y):
    """Compute the pairwise Euclidean similarity between x and y.
    This function computes the following similarity value between each pair of x_i
    and y_j: s(x_i, y_j) = -|x_i - y_j|^2.
    Args:
      x: NxD float tensor.
      y: MxD float tensor.
    Returns:
      s: NxM float tensor, the pairwise euclidean similarity.
    """
    s = 2 * torch.mm(x, torch.transpose(y, 1, 0))
    diag_x = torch.sum(x * x, dim=-1)
    diag_x = torch.unsqueeze(diag_x, 0)
    diag_y = torch.reshape(torch.sum(y * y, dim=-1), (1, -1))

    return s - diag_x - diag_y


def pairwise_dot_product_similarity(x, y):
    """Compute the dot product similarity between x and y.
    This function computes the following similarity value between each pair of x_i
    and y_j: s(x_i, y_j) = x_i^T y_j.
    Args:
      x: NxD float tensor.
      y: MxD float tensor.
    Returns:
      s: NxM float tensor, the pairwise dot product similarity.
    """
    return torch.mm(x, torch.transpose(y, 1, 0))


def pairwise_cosine_similarity(x, y):
    """Compute the cosine similarity between x and y.
    This function computes the following similarity value between each pair of x_i
    and y_j: s(x_i, y_j) = x_i^T y_j / (|x_i||y_j|).
    Args:
      x: NxD float tensor.
      y: MxD float tensor.
    Returns:
      s: NxM float tensor, the pairwise cosine similarity.
    """
    x = torch.div(x, torch.sqrt(torch.max(torch.sum(x ** 2), 1e-12)))
    y = torch.div(y, torch.sqrt(torch.max(torch.sum(y ** 2), 1e-12)))
    return torch.mm(x, torch.transpose(y, 1, 0))


PAIRWISE_SIMILARITY_FUNCTION = {
    'euclidean': pairwise_euclidean_similarity,
    'dotproduct': pairwise_dot_product_similarity,
    'cosine': pairwise_cosine_similarity,
}


def get_pairwise_similarity(name):
    """Get pairwise similarity metric by name.
    Args:
      name: string, name of the similarity metric, one of {dot-product, cosine,
        euclidean}.
    Returns:
      similarity: a (x, y) -> sim function.
    Raises:
      ValueError: if name is not supported.
    """
    if name not in PAIRWISE_SIMILARITY_FUNCTION:
        raise ValueError('Similarity metric name "%s" not supported.' % name)
    else:
        return PAIRWISE_SIMILARITY_FUNCTION[name]


def compute_cross_attention(x, y, sim):
    """Compute cross attention.
    x_i attend to y_j:
    a_{i->j} = exp(sim(x_i, y_j)) / sum_j exp(sim(x_i, y_j))
    y_j attend to x_i:
    a_{j->i} = exp(sim(x_i, y_j)) / sum_i exp(sim(x_i, y_j))
    attention_x = sum_j a_{i->j} y_j
    attention_y = sum_i a_{j->i} x_i
    Args:
      x: NxD float tensor.
      y: MxD float tensor.
      sim: a (x, y) -> similarity function.
    Returns:
      attention_x: NxD float tensor.
      attention_y: NxD float tensor.
    """
    a = sim(x, y)
    a_x = torch.softmax(a, dim=1)  # i->j
    a_y = torch.softmax(a, dim=0)  # j->i
    attention_x = torch.mm(a_x, y)
    attention_y = torch.mm(torch.transpose(a_y, 1, 0), x)
    return attention_x, attention_y


def pair_attention(x: Tensor, x_batch: Tensor,
                   y: Tensor, y_batch: Tensor,
                   similarity='dotproduct'):
    assert (x_batch.unique() == y_batch.unique()).all()

    sim = get_pairwise_similarity(similarity)

    # This is probably better than doing boolean_mask for each i
    pairs = []
    for i in x_batch.unique():
        pairs.append((x[x_batch == i, :], y[y_batch == i, :]))

    attention_xs = []
    attention_ys = []

    for x, y in pairs:
        attention_x, attention_y = compute_cross_attention(x, y, sim)
        attention_xs.append(attention_x)
        attention_ys.append(attention_y)

    return torch.cat(attention_xs, dim=0), torch.cat(attention_ys, dim=0)


class GraphMatchingLayer(MessagePassing):


    def __init__(self,
                node_dim: int,
                hidden_dims: List[int],
                node_update_type: str = "residual",
                layer_norm: bool = False,
                similarity: str = 'dotproduct',
                **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        assert similarity in PAIRWISE_SIMILARITY_FUNCTION
        self.similarity = similarity
        assert node_update_type in ["mlp", "residual"]
        self.node_update_type = node_update_type
        self.layer_norm = layer_norm
        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm()
            self.layer_norm2 = nn.LayerNorm()

        layer = []
        layer.append(nn.Linear(2 * node_dim, hidden_dims[0]))
        for i in range(1, len(hidden_dims)):
            layer.append(nn.ReLU())
            layer.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
        self.message_net = nn.Sequential(*layer)

        layer = []
        layer.append(nn.Linear(3 * hidden_dims[0], hidden_dims[0]))
        for i in range(1, len(hidden_dims)):
            layer.append(nn.ReLU())
            layer.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
        self.update_net = nn.Sequential(*layer)


        layer = []
        layer.append(nn.Linear(node_dim, hidden_dims[0]))
        for i in range(1, len(hidden_dims)):
            layer.append(nn.ReLU())
            layer.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
        self.enc = nn.Sequential(*layer)


    def forward(self,
                x: Tensor, x_edge_index: Tensor, x_batch: Tensor,
                y: Tensor, y_edge_index: Tensor, y_batch: Tensor,
                x_size: Size = None, y_size: Size = None):

        def propagate(edge_index, x, size):
            node_states: Tensor = self.propagate(edge_index, x=x, size=size)
            if self.layer_norm:
                node_states = self.layer_norm1(node_states)
            return node_states

        x_node_states: Tensor = propagate(x_edge_index, x, x_size)
        y_node_states: Tensor = propagate(y_edge_index, y, y_size)

        x = self.enc(x)
        y = self.enc(y)
        attention_xs, attention_ys = pair_attention(x, x_batch, y, y_batch,
                                                    similarity=self.similarity)
        attended_xs = x_node_states - attention_xs
        attended_ys = y_node_states - attention_ys


        def update(x, node_states, attention):
            xs = torch.cat((x, node_states, attention), dim=-1)
            mlp_output = self.update_net(xs)
            if self.layer_norm:
                mlp_output = self.layer_norm2(mlp_output)
            if self.node_update_type == 'mlp':
                return mlp_output
            return x + mlp_output

        x = update(x, x_node_states, attended_xs)
        y = update(y, y_node_states, attended_ys)

        return x, y


    def message(self, x_j: Tensor, edge_index: Tensor) -> Tensor:
        from_idx = edge_index[0, :]
        to_idx = edge_index[1, :]
        from_states = x_j[from_idx]
        to_states = x_j[to_idx]

        edge_inputs = torch.cat([from_states, to_states], dim=-1)
        return self.message_net(edge_inputs)


class GINMatchingLayer(MessagePassing):

    def __init__(self, node_dim: int, hidden_dims: List[int]):
        super().__init__()
        layer = []
        layer.append(nn.Linear(node_dim, hidden_dims[0]))
        for i in range(1, len(hidden_dims)):
            layer.append(nn.ReLU())
            layer.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
        self.gin = GINConv(nn.Sequential(*layer))

        layer = []
        layer.append(nn.Linear(node_dim, hidden_dims[0]))
        for i in range(1, len(hidden_dims)):
            layer.append(nn.ReLU())
            layer.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
        self.enc = nn.Sequential(*layer)

        layer = []
        layer.append(nn.Linear(3 * hidden_dims[0], hidden_dims[0]))
        for i in range(1, len(hidden_dims)):
            layer.append(nn.ReLU())
            layer.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
        self.update_net = nn.Sequential(*layer)


    def forward(self, x, x_edge_index, x_batch, y, y_edge_index, y_batch, x_size=None, y_size=None):

        x_node_states = self.gin.forward(x, x_edge_index, x_size)
        y_node_states = self.gin.forward(y, y_edge_index, y_size)

        x = self.enc(x)
        y = self.enc(y)
        attention_xs, attention_ys = pair_attention(x, x_batch, y, y_batch)

        attended_xs = x_node_states - attention_xs
        attended_ys = y_node_states - attention_ys

        def update(x, node_states, attention):
            xs = torch.cat((x, node_states, attention), dim=-1)
            mlp_output = self.update_net(xs)
            return x + mlp_output

        x = update(x, x_node_states, attended_xs)
        y = update(y, y_node_states, attended_ys)

        return x, y

if __name__ == "__main__":
    gmn = GraphMatchingLayer(2, [4])
    x = torch.rand((3, 2))
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 0],
        [1, 0, 2, 1, 0, 2]
    ]).long()
    batch = torch.tensor([0, 0, 0]).long()

    out = gmn(x, edge_index, batch, x - 1, edge_index, batch)
    print(out[0])
    print(out[1])
