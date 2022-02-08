from itertools import product, chain
from pathlib import Path
from sys import argv
from typing import List, Literal, Tuple
import numpy as np

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import average_precision_score, f1_score
from tqdm import tqdm

from utils.data_module import PlagiarismFunctions
from utils.lightning_module import PlagiarismModel

seed_everything(420)


graph_architecture_sweep = {
    "embedding_dim": [4],
    "classify": [False, True],
    "hidden_dim": [64, 128, 256, 512],
    "num_layers": [1, 3, 5],
    "layer_type": ["gin"]
}

tokens_architecture_sweep = {
    "embedding_dim": [8],
    "classify": [False, True],
    "hidden_dim": [256, 128, 64],
    "num_layers": [3, 2, 1]
}

cache = {}
data_cache = "data/cache"


def best_f1(labels, probs) -> float:

    best_f1 = 0
    for thr in np.linspace(min(probs), max(probs), 1001):
        f1 = f1_score(labels, probs > thr)
        if f1 > best_f1:
            best_f1 = f1

    return best_f1

@torch.no_grad()
def validate(model: PlagiarismModel, test_dataloader: DataLoader) -> Tuple[float, float]:
    similarities = []
    labels = []
    model.cuda().eval()
    for batch in tqdm(test_dataloader, desc="Validating..."):
        a, p, n, la, *_ = batch
        sims = model.similarity(a.cuda(), p.cuda()).cpu().numpy()
        similarities.append(sims)
        labels.extend([1] * len(la))
        sims = model.similarity(a.cuda(), n.cuda()).cpu().numpy()
        similarities.append(sims)
        labels.extend([0] * len(la))

    similarities = np.concatenate(similarities)
    labels = np.array(labels)
    return average_precision_score(labels, similarities), best_f1(labels, similarities)

def kfold_cross_validation(functions: List[str], model_params: dict, data_params: dict, k: int = 5) -> Tuple[float, float]:
    if k < 0:
        model_type = "gnn" if data_params["format"] == "graph" else "lstm"
        split_len = len(functions)
        val_split = set(functions[int(0.8 * split_len):])
        train_split = list(set(functions) - val_split)
        data_module = PlagiarismFunctions(model_params["classify"], train_split, list(val_split), **data_params)
        data_module.setup()
        # print(len(data_module.train_dataset))

        model = PlagiarismModel(**model_params, model_type=model_type, num_tokens=data_module.num_features)
        checkpoint_callback = ModelCheckpoint(monitor='loss/val', save_top_k=1)
        early_stopping = EarlyStopping("loss/val", patience=10)
        trainer = Trainer(gpus=1, max_epochs=100, callbacks=[checkpoint_callback, early_stopping], num_sanity_val_steps=0)

        trainer.fit(model, train_dataloader=data_module.train_dataloader(), val_dataloaders=data_module.val_dataloader())
        model = PlagiarismModel.load_from_checkpoint(checkpoint_callback.best_model_path)

        ap, f1 = validate(model, data_module.test_dataloader())
        print("AP:", ap)
        print("F1:", f1)
        return ap, f1
    else:
        split_len = len(functions) // k
        metrics = []
        model_type = "gnn" if data_params["format"] == "graph" else "lstm"
        for i in range(k):
            val_split = set(functions[i * split_len: (i + 1) * split_len])
            train_split = list(set(functions) - val_split)
            data_module = PlagiarismFunctions(model_params["classify"], train_split, list(val_split), **data_params)
            data_module.setup()

            model = PlagiarismModel(**model_params, model_type=model_type, num_tokens=data_module.num_features)
            checkpoint_callback = ModelCheckpoint(monitor='loss/val', save_top_k=1)
            early_stopping = EarlyStopping("loss/val", patience=6)
            trainer = Trainer(gpus=1, max_epochs=20, callbacks=[checkpoint_callback, early_stopping], num_sanity_val_steps=0)

            trainer.fit(model, train_dataloader=data_module.train_dataloader(), val_dataloaders=data_module.val_dataloader())
            model = PlagiarismModel.load_from_checkpoint(checkpoint_callback.best_model_path)

            metrics.append(validate(model, data_module.test_dataloader()))
            print(metrics[-1])

        return np.mean(metrics), np.std(metrics)

def main(format: Literal["graph", "tokens"]):
    if format == "tokens":
        data_root = Path("data/functions")
        params_to_sweep = tokens_architecture_sweep 
        batch_size = 50
    else:
        data_root = Path("data/graph_functions")
        params_to_sweep = graph_architecture_sweep
        batch_size = 200
    res_file = Path(f"results_{format}.txt")
    # if res_file.exists():
    #     res_file.unlink()
    # train_split = data_root / "train.txt"
    train_functions = []
    for split_file in [data_root / "train.txt", data_root / "val.txt"]:
        with split_file.open() as f:
            train_functions += list(map(lambda l: l.strip(), f.readlines()))

    for param_values in product(*params_to_sweep.values()):
        model_params = dict(zip(params_to_sweep.keys(), param_values))
        data_params = {
            "root_dir": str(data_root),
            "format": format,
            "cache": data_cache,
            "batch_size": batch_size
        }
        all_params_strs = list(map(str, chain(data_params.values(), model_params.values())))
        # if res_file.exists():
        #     with res_file.open() as f:
        #         if any(",".join(map(str, model_params.values())) in line for line in f):
        #             print("Skipping...")
        #             continue

        ap, f1 = kfold_cross_validation(train_functions, model_params, data_params, -1)
        with res_file.open("a") as f:
            f.write(",".join(all_params_strs + [str(ap), str(f1)]) + "\n")


if __name__ == "__main__":
    main(argv[1])
