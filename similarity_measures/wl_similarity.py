from multiprocessing import Pool
import torch
from torch_geometric.nn import WLConv
from torch_geometric.data import Batch
from sklearn.preprocessing import OneHotEncoder

from datasets.utils.graph_attrs import node_attrs

def similarity(hist1, hist2):
    return 1 - (torch.abs(hist1 - hist2).sum(dim=1) / (hist1.sum(dim=1) + hist2.sum(dim=1)))

def distance(hists1, hists2):
    return torch.abs(hists1 - hists2).sum(dim=1)

def num_nodes(graph: Batch):
    _, counts = torch.unique(graph.batch, return_counts=True, sorted=True)
    return counts

def distance_single(hists1, hists2):
    return torch.abs(hists1 - hists2).sum()

def body(f1, f2, num_layers):
    convs = torch.nn.ModuleList([WLConv() for _ in range(num_layers)])
    enc = OneHotEncoder(sparse=False)
    enc.fit([[i] for i, label in enumerate(node_attrs)])
    d = 0
    f1.x = torch.tensor(enc.transform(f1.x.numpy()))
    f2.x = torch.tensor(enc.transform(f2.x.numpy()))
    for conv in convs:
        f1.x = conv(f1.x, f1.edge_index)
        f2.x = conv(f2.x, f2.edge_index)
        d += distance_single(conv.histogram(f1.x, norm=False),
                             conv.histogram(f2.x, norm=False))
    sim = 1 - d / (num_layers * (len(f1.x) + len(f2.x)))
    return sim.item()



class WL(torch.nn.Module):
    def __init__(self, num_layers):
        super(WL, self).__init__()
        self.pool = Pool()
        self.num_layers = num_layers
        # self.convs = torch.nn.ModuleList([WLConv() for _ in range(num_layers)])
        # self.enc = OneHotEncoder(sparse=False)
        # self.enc.fit([[i] for i, label in enumerate(node_attrs)])

    def forward(self, f1: Batch, f2: Batch):
        handles = [self.pool.apply_async(body, (x, y, self.num_layers)) for x, y in zip(f1.to_data_list(), f2.to_data_list())]
        return [h.get() for h in handles]
        # lens_f1 = num_nodes(f1)
        # lens_f2 = num_nodes(f2)
        # d = 0
        # f1.x = torch.tensor(self.enc.transform(f1.x.numpy()))
        # f2.x = torch.tensor(self.enc.transform(f2.x.numpy()))
        # for conv in self.convs:
        #     f1.x = conv(f1.x, f1.edge_index)
        #     f2.x = conv(f2.x, f2.edge_index)
        #     d += distance(conv.histogram(f1.x, f1.batch, norm=False),
        #                   conv.histogram(f2.x, f2.batch, norm=False))
        # sim = 1 - d / (len(self.convs) * torch.minimum(lens_f1, lens_f2))
        # return sim
