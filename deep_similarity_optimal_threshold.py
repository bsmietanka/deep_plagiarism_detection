from functools import partial
import json
from os import path
from typing import Callable, Union
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm
from torch import LongTensor
from torch_geometric.data import Data
from sklearn.metrics import f1_score

from datasets.functions_dataset import FunctionsDataset
from similarity_measures import DeepSimilarity


RepresentationType = Union[LongTensor, Data]
MeasureType = Callable[[RepresentationType, RepresentationType], float]


configs_by_format = {
    "graph": [
    #     # "exp_pyl.json",
    #     # "exp_cont.json",
        "config.json"
    ],
    "tokens": [
        "config_lstm.json"
    ],
    # "letters": [
    #     "configs/letters/config.json"
    # ]
}

root_dir_for_formats = {
    "tokens": "data/functions",
    "graph": "data/graph_functions"
}


def calibrate_threshold(measure: MeasureType, val_pairs, val_labels):

    val_similarities = []
    for pairs in tqdm(val_pairs):
        similarities = []
        for pair in pairs:
            similarities.append(measure(*pair))
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
    for thr in tqdm(np.linspace(0., 1., 1001)):
        f1, std = objective_function(thr, val_similarities, val_labels)
        if f1 > best_f1:
            best_f1 = f1
            best_std = std
            best_thr = thr

    return best_thr, best_f1, best_std


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


def main(n_funs_per_split: int = 20, multiplier: int = 4, dataset_args: dict = dict()):
    assert dataset_args["mode"] == "triplets"

    results = []
    for format, measure_configs in configs_by_format.items():

        dataset_args["format"] = format
        dataset_args["root_dir"] = root_dir_for_formats[format]
        dataset = FunctionsDataset(**dataset_args)
        num_funs = dataset.num_functions()

        val_datasets = [FunctionsDataset(**dataset_args, split_subset=(i * n_funs_per_split, (i + 1) * n_funs_per_split))
                        for i in range(num_funs // n_funs_per_split)]

        worker = partial(prepare_dataset, multiplier=multiplier)

        with Pool() as pool:
            val_pairs_labels = list(tqdm(
                    pool.imap(worker, val_datasets),
                    total=len(val_datasets),
                    desc=f"Preparing {format} dataset"
                    ))
        val_pairs, val_labels = list(map(list, zip(*val_pairs_labels)))


        for config_path in measure_configs:

            with open(config_path, "r") as f:
                config_json = json.load(f)
            weights_path = path.join("runs", config_path.replace(".json", ""), "models/best.pt")
            measure = DeepSimilarity(model_type=config_json["encoder_type"],dataset=dataset, device="cuda:0", weights=weights_path, **config_json["encoder"])
            opt_thr, f1_opt, f1_std = calibrate_threshold(measure, val_pairs, val_labels)

            partial_res = {
                "measure": "deep_similarity",
                "args": config_path,
                "best_thr": opt_thr,
                "best_avg_f1": f1_opt,
                "f1_std": f1_std
            }
            print(partial_res)
            results.append(partial_res)

    return results


if __name__ == "__main__":
    dataset_args_base = {
        "split_file": "val.txt",
        "mode": "triplets",
        "return_tensor": True
    }
    results = main(n_funs_per_split=50, dataset_args=dataset_args_base)

    with open('deep_thresholds.json', 'w') as outfile:
        json.dump(results, outfile)

