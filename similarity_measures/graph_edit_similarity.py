from typing import Union
from networkx.algorithms.similarity import graph_edit_distance
from networkx import Graph
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

import gmatch4py as gm


GraphType = Union[Data, Graph]

class GraphEditSimilarity:

    def graph_size(self, g: Graph):
        return g.number_of_edges() + g.number_of_nodes()

    def __call__(self, g1: GraphType, g2: GraphType, roots=(0,0)) -> float:
        if isinstance(g1, Data):
            g1 = to_networkx(g1)
        if isinstance(g2, Data):
            g2 = to_networkx(g2)
        # ged = graph_edit_distance(g1, g2, roots=roots)

        # return 1 - ged / max(self.graph_size(g1), self.graph_size(g2))
        ged = gm.GraphEditDistance(1, 1, 1, 1)

        res = ged.compare([g1, g2], None)
        return ged.similarity(res)[0,1]


if __name__ == "__main__":
    ed = GraphEditSimilarity()

    import networkx as nx
    g1 = nx.balanced_tree(2, 4)
    g2 = nx.balanced_tree(2, 4)
    print(ed(g1, g2, (0, 1)))
