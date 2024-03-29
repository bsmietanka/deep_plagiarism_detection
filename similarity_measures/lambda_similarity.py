
from typing import Optional, Union

import netcomp as nc
from networkx import Graph, adjacency_matrix
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

GraphType = Union[Data, Graph]

class LambdaSimilarity:

    def __init__(self, k: Optional[int] = 10):
        self.k = k

    def __call__(self, g1: GraphType, g2: GraphType):
        if isinstance(g1, Data):
            g1 = to_networkx(g1)
        if isinstance(g2, Data):
            g2 = to_networkx(g2)
        a1 = adjacency_matrix(g1)
        a2 = adjacency_matrix(g2)
        x = nc.lambda_dist(a1, a2, kind='laplacian_norm', k=self.k)
        min_size = min(a1.shape[0], a2.shape[0])
        if self.k is not None and self.k > 0 and self.k < min_size:
            return 1 - x / self.k
        return 1 - x / min_size


if __name__ == "__main__":
    ls = LambdaSimilarity(50)

    import networkx as nx
    g1 = nx.star_graph(100)
    g2 = nx.path_graph(100)
    print(ls(g1, g2))
