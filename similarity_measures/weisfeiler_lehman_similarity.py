from typing import Union
import gmatch4py as gm
from networkx import Graph
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx



GraphType = Union[Data, Graph]

class WeisfeilerLehmanSimilarity:

    def __init__(self, h: int = 2):
        self.wlk = gm.WeisfeilerLehmanKernel(h)


    def __call__(self, g1: GraphType, g2: GraphType) -> float:
        if isinstance(g1, Data):
            g1 = to_networkx(g1)
        if isinstance(g2, Data):
            g2 = to_networkx(g2)

        res = self.wlk.compare([g1, g2], None)
        return self.wlk.similarity(res)[0,1]


if __name__ == "__main__":
    wls = WeisfeilerLehmanSimilarity()

    import networkx as nx
    g1 = nx.balanced_tree(2, 4)
    g2 = nx.star_graph(5)
    print(wls(g1, g2))
