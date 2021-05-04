from datasets.functions_dataset import FunctionsDataset
from tqdm import tqdm
from multiprocessing import Pool
from torch_geometric.utils import get_laplacian, contains_self_loops, remove_self_loops

dataset = FunctionsDataset("data/graph_functions/", "train.txt", "singles", "graph", cache="data/cache")

subset = list(range(len(dataset)))[:10000]
with Pool() as p:
    graphs = list(tqdm(p.imap(dataset.__getitem__, subset, chunksize=256), total=len(subset)))

graphs = [row[0] for row in graphs]

self_loops = [contains_self_loops(g.edge_index) for g in graphs]

print("num self loops", self_loops.count(True))
print("average num nodes", sum(g.num_nodes for g in graphs) / len(graphs))
print("average num edges", sum(g.num_edges for g in graphs) / len(graphs))

def remove(g):
    g.edge_index = remove_self_loops(g.edge_index)[0]
    return g

without_self_loops = [remove(g) for g in graphs]
print("average num edges without self loops", sum(g.num_edges for g in without_self_loops) / len(without_self_loops))
