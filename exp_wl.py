from functools import partial
from pprint import pprint
from typing import Union
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.nn import WLConv
from torch_geometric.data import Data, Batch
from sklearn.preprocessing import OneHotEncoder

from datasets.utils.graph_attrs import node_attrs
from datasets.functions_dataset import FunctionsDataset


RepresentationType = Union[Data]

measures_args = {
    "wlk": [1, 2, 3],
}

class WL(torch.nn.Module):
    def __init__(self, num_layers):
        super(WL, self).__init__()
        self.convs = torch.nn.ModuleList([WLConv() for _ in range(num_layers)])
        self.enc = OneHotEncoder(sparse=False)
        self.enc.fit([[i] for i, label in enumerate(node_attrs)])

    def forward(self, x, edge_index, batch=None):
        hists = []
        x = torch.tensor(self.enc.transform(x.numpy()))
        for conv in self.convs:
            x = conv(x, edge_index)
            hists.append(conv.histogram(x, batch, norm=False))
        return hists


class Worker:
    def __init__(self, num_iter):
        self.num_iter = num_iter
        self.wl = WL(num_iter)

    def __call__(self, f1f2):
        f1, f2 = f1f2
        data = Batch.from_data_list([f1, f2])
        hists = self.wl(data.x, data.edge_index, data.batch)
        sims = []
        for hist in hists:
            print(torch.min(hist, 0)[0])
            sim = 1 - (torch.abs(hist[0] - hist[1]).sum() / (hist[0].sum() + hist[1].sum()))
            sims.append(sim.item())
        return np.mean(sims)


def prepare_dataset(val_dataset, multiplier=4):
    pairs = []
    labels = []
    for i in range(multiplier * len(val_dataset)):
        a, p, n, _ = val_dataset[i % len(val_dataset)]

        pairs.append((a, p))
        labels.append(1)
        pairs.append((a, n))
        labels.append(0)
    return pairs, np.array(labels)


def calibrate_threshold(num_iters, val_pairs, val_labels):

    val_similarities = []
    for pairs in tqdm(val_pairs):
        worker = Worker(num_iters)
        with Pool() as pool:
            similarities = list(pool.imap(worker, pairs))
        val_similarities.append(np.array(similarities))

    def objective_function(thr, val_similarities, val_labels):
        assert len(val_similarities) == len(val_labels)
        f1s = []
        for similarities, labels in zip(val_similarities, val_labels):
            f1s.append(f1_score(labels, similarities > thr))
        return np.mean(f1s), np.std(f1s)

    roc_auc = []
    for sim, lab in zip(val_similarities, val_labels):
        roc_auc.append(roc_auc_score(lab, sim))
    print("ROC AUC:", np.mean(roc_auc), f"({np.std(roc_auc)})")

    best_f1 = 0
    best_thr = 0
    best_std = 0
    for thr in np.linspace(0., 1., 1001):
        f1, std = objective_function(thr, val_similarities, val_labels)
        if f1 > best_f1:
            best_f1 = f1
            best_std = std
            best_thr = thr

    return best_thr, best_f1, best_std


def main(n_funs_per_split: int = 20, multiplier: int = 10, dataset_args: dict = dict()):

    assert dataset_args["mode"] == "triplets"

    dataset = FunctionsDataset(**dataset_args)
    num_funs = dataset.num_functions()


    results = []
    val_datasets = [FunctionsDataset(**dataset_args, split_subset=(i * n_funs_per_split, (i + 1) * n_funs_per_split))
                    for i in range(num_funs // n_funs_per_split)]

    worker = partial(prepare_dataset, multiplier=multiplier)

    with Pool() as pool:
        val_pairs_labels = list(tqdm(
                pool.imap(worker, val_datasets),
                total=len(val_datasets),
                desc=f"Preparing dataset"
                ))
    val_pairs, val_labels = list(map(list, zip(*val_pairs_labels)))

    for num_iter in measures_args["wlk"]:
        opt_thr, f1_opt, f1_std = calibrate_threshold(num_iter, val_pairs, val_labels)

        partial_res = {
            "measure": "wl",
            "num_iter": num_iter,
            "best_thr": opt_thr,
            "best_avg_f1": f1_opt,
            "f1_std": f1_std
        }
        print(partial_res)
        results.append(partial_res)

    return results



if __name__ == "__main__":
    dataset_args_base = {
        "root_dir": "data/graph_functions",
        "split_file": "val.txt",
        "mode": "triplets",
        "format": "graph",
        "return_tensor": True
    }
    results = main(n_funs_per_split=50, dataset_args=dataset_args_base)

    pprint(results)

