from functools import partial
from pprint import pprint
from typing import Union
from multiprocessing import Pool

import igraph as ig
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch_geometric.data import Data
from wwl import wwl

from datasets.functions_dataset import FunctionsDataset


RepresentationType = Union[Data]

measures_args = {
    "wlk": [2, 5, 9, 13, 17, 20],
}

def pyg2ig(g: Data) -> ig.Graph:
    ea = g.edge_attr.flatten().tolist()
    ea = {"label": ea}
    va = g.x.flatten().tolist()
    va = {"label": va}
    ei = g.edge_index.t().tolist()
    return ig.Graph(edges=ei, edge_attrs=ea, vertex_attrs=va)

class Worker:
    def __init__(self, num_iter):
        self.num_iter = num_iter

    def __call__(self, f1f2):
        f1, f2 = f1f2
        f1 = pyg2ig(f1)
        f2 = pyg2ig(f2)
        similarity = wwl([f1, f2], num_iterations=self.num_iter)
        return similarity[0, 1]


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


def main(n_funs_per_split: int = 20, multiplier: int = 4, dataset_args: dict = dict()):

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
            "measure": "wlk",
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

