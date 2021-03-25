from itertools import combinations
import json
from typing import Callable, List, Union
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from networkx import Graph

from datasets.functions_dataset import FunctionsDataset
from similarity_measures import *
from datasets.utils.r_tokens import r_tokens


RepresentationType = Union[str, List[int], Graph]
MeasureType = Callable[[RepresentationType, RepresentationType], float]


measures_by_format = {
    "tokens": ["edit", "gst"],
    "graph": ["graph_edit", "lambda", "wlk"]
    }

measures_args = {
    "edit": [{}],
    "gst": [{"min_len": val} for val in range(3, 20, 2)],
    "graph_edit": [{}],
    "lambda": [{"k": val} for val in [10, 30, 100, None]],
    "wlk": [{"h": val} for val in [2, 5, 9, 13, 17, 20]]
}


def get_measure(method: str, **method_args) -> MeasureType:
    if method == "edit":
        return EditSimilarity(**method_args)
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


class Worker:
    def __init__(self, measure: MeasureType):
        self.measure = measure

    def __call__(self, f1f2):
        f1, f2 = f1f2

        similarity = self.measure(f1, f2)
        return similarity


def calibrate_threshold(measure: MeasureType, val_pairs, val_labels):

    with Pool(16) as pool:
        val_similarities, val_labels = [], []
        for pairs in val_pairs:
            worker = Worker(measure)
            similarities = np.array(tqdm(pool.imap(worker, pairs, chunksize=5), total=len(pairs)))
            val_similarities.append(similarities)

    # cache results? takes shitload of time

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
    for format, measures in measures_by_format.items():

        dataset_args["format"] = format

        val_datasets = [FunctionsDataset(**dataset_args, split_subset=(i * n_funs_per_split, (i + 1) * n_funs_per_split))
                        for i in range(num_funs // n_funs_per_split)]

        val_pairs = []
        val_labels = []
        for val_dataset in tqdm(val_datasets, desc=f"Preparing {format} datasets"):
            pairs = []
            labels = []
            for i in range(multiplier * len(val_dataset)):
                a, p, n, _ = val_dataset[i % len(val_dataset)]
                pairs.append((a, p))
                labels.append(1)
                pairs.append((a, n))
                labels.append(0)
            val_pairs.append(pairs)
            val_labels.append(np.array(labels))


        for method in measures:

            for args in measures_args[method]:
                measure = get_measure(method, **args)
                opt_thr, f1_opt, f1_std = calibrate_threshold(measure, val_pairs, val_labels)

                results.append({
                    "measure": method,
                    "args": args,
                    "best_thr": opt_thr,
                    "best_avg_f1": f1_opt,
                    "f1_std": f1_std
                })

    return results



if __name__ == "__main__":
    dataset_args_base = {
        "root_dir": "data/functions",
        "split_file": "train.txt",
        "mode": "triplets",
        "format": "tokens",
        "return_tensor": False
    }
    results = main("edit", n_funs_per_split=50, dataset_args=dataset_args_base)

    with open('thresholds.json', 'w') as outfile:
        json.dump(results, outfile)

