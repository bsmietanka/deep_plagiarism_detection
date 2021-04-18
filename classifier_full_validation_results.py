import json
from os import path
from typing import Callable, List, Union
from utils.train_utils import get_model

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from torch import LongTensor
from torch_geometric.data import Data
from sklearn.metrics import f1_score

from datasets.functions_dataset import FunctionsDataset
from datasets.utils.collate import Collater


RepresentationType = Union[LongTensor, Data]
MeasureType = Callable[[RepresentationType, RepresentationType], float]


configs_by_format = {
    "graph": [
        "configs/graphs/config.json"
    ],
    # "tokens": [
    #     "configs/tokens/config.json"
    # ],
    # "letters": [
    #     "configs/letters/config.json"
    # ]
}

root_dir_for_formats = {
    "tokens": "data/functions",
    "graph": "data/graph_functions"
}


@torch.no_grad()
def calibrate_threshold(model, val_dataloaders: List[DataLoader], multiplier: int = 4):

    val_similarities = []
    val_labels = []
    for val_dataloader in tqdm(val_dataloaders):
        similarities = []
        labels = []
        for i in range(multiplier):
            for a, p, n, _ in val_dataloader:
                a = a.to("cuda:0")
                pred_p = model.classify(a, p.to("cuda:0")).cpu().numpy()
                pred_n = model.classify(a, n.to("cuda:0")).cpu().numpy()
                similarities.extend([pred_p, pred_n])
                labels.extend([np.ones_like(pred_p), np.zeros_like(pred_n)])
        similarities = np.concatenate(similarities).flatten()
        labels = np.concatenate(labels).flatten()
        val_similarities.append(similarities)
        val_labels.append(labels)


    def objective_function(thr, val_similarities, val_labels):
        assert len(val_similarities) == len(val_labels)
        f1s = []
        for similarities, labels in zip(val_similarities, val_labels):
            f1s.append(f1_score(labels, similarities > thr))
        return np.mean(f1s), np.std(f1s)

    best_f1 = 0
    best_thr = 0
    best_std = 0
    for thr in tqdm(np.linspace(0., 1., 1001), desc="Finding optimal threshold"):
        f1, std = objective_function(thr, val_similarities, val_labels)
        if f1 > best_f1:
            best_f1 = f1
            best_std = std
            best_thr = thr

    return best_thr, best_f1, best_std


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
        val_dataloaders = [DataLoader(d, batch_size=512, num_workers=10, collate_fn=Collater(), pin_memory=True) for d in val_datasets]


        for config_path in measure_configs:

            with open(config_path, "r") as f:
                config_json = json.load(f)
            weights_path = path.join("runs", config_path.replace(".json", ""), "models/best.pt")

            encoder = get_model(config_json["encoder_type"], dataset, **config_json["encoder"])

            opt_thr, f1_opt, f1_std = calibrate_threshold(encoder, val_dataloaders, multiplier=multiplier)

            partial_res = {
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
    results = main(n_funs_per_split=40, dataset_args=dataset_args_base)
