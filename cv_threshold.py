from itertools import combinations
from typing import Callable, List, Optional, Union
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from scipy.optimize import fmin
from networkx import Graph

from datasets.functions_dataset import FunctionsDataset
from similarity_measures import *


RepresentationType = Union[str, List[int], Graph]
MeasureType = Callable[[RepresentationType, RepresentationType], float]


def get_measure(method: str, **method_args) -> MeasureType:
    if method == "edit":
        return EditSimilarity()
    elif method == "gst":
        return GSTSimilarity(**method_args)
    elif method == "graph_edit":
        return GraphEditSimilarity()
    elif method == "lambda":
        return LambdaSimilarity(**method_args)
    elif method == "wlk":
        return WeisfeilerLehmanSimilarity(**method_args)
    else:
        raise ValueError(f"Unsupported similarity measure: {method}")



def calibrate_threshold(measure: MeasureType, n_funs_per_split: int = 20, **dataset_args):
    
    assert dataset_args["mode"] == "singles"

    if isinstance(measure, (EditSimilarity, GSTSimilarity)):
        assert dataset_args["format"] in ["tokens", "letters"]
    else:
        assert dataset_args["format"] == "graph"

    dataset = FunctionsDataset(**dataset_args)
    num_funs = dataset.num_functions()

    val_datasets = [FunctionsDataset(**dataset_args, split_subset=(i * n_funs_per_split, (i + 1) * n_funs_per_split))
                        for i in range(num_funs // n_funs_per_split)]

    class Worker:
        def __init__(self, measure: MeasureType, dataset: FunctionsDataset):
            self.measure = measure
            self.dataset = dataset

        def __call__(self, idx_pair):
            first, second = idx_pair
            f1body, f1name = self.dataset[first]
            f2body, f2name = self.dataset[second]

            similarity = measure(f1body, f2body)
            label = f1name == f2name
            return similarity, label

    val_similarities, val_labels = [], []
    for val_dataset in val_datasets:
        worker = Worker(measure, val_dataset)
        all_pairs = list(combinations(range(len(val_dataset)), 2))
        with Pool(16) as p:
            similarities_labels = list(tqdm(p.imap(worker, all_pairs), total=len(all_pairs)))
        similarities, labels = map(np.array, zip(*similarities_labels))
        val_similarities.append(similarities)
        val_labels.append(labels)

    def objective_function(thr, val_similarities, val_labels):
        assert len(val_similarities) == len(val_labels)
        f1s = []
        for similarities, labels in zip(val_similarities, val_labels):
            f1s.append(f1_score(labels, similarities > thr)))
        return - np.mean(f1s)

    opt_thr = fmin(objective_function, x0=0.5, args=(val_similarities, val_labels))

    return opt_thr


def main(method: str, n_funs_per_split: int = 20, method_args: dict = dict(), dataset_args: dict = dict()):

    measure = get_measure(method, **method_args)

    opt_thr = calibrate_threshold(measure, n_funs_per_split, **dataset_args)

    return opt_thr


if __name__ == "__main__":
    dataset_args_base = {
        "data_root": "data/functions",
        "split_file": "train.csv",
        "mode": "singles",
        "format": "tokens",
        "return_tensor": False
    }
    ot = main("edit", dataset_args=dataset_args_base)

    print(ot)
