
import json
from typing import Union
from pprint import pprint
from utils.train_utils import create_triplet_dataset, get_model, n_triplet_dataset


import torch
from torch import nn
from pytorch_metric_learning import samplers
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import RandomSampler
from tqdm import tqdm

from datasets.utils.collate import Collater
from datasets.functions_dataset import FunctionsDataset
from encoders.classifier import Classifier

from pytorch_lightning import seed_everything
seed_everything(42)



def train_epoch(model: Union[nn.Module, Classifier],
               train_loader: DataLoader,
               loss_fun: nn.Module,
               optimizer: optim.Optimizer,
               device: str):
    model.train()

    epoch_loss = 0.
    optimizer.zero_grad()
    for a, p, n, *_ in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()

        a = a.to(device)
        p = p.to(device)
        n = n.to(device)
        pos = model.classify(a, p)
        neg = model.classify(a, n)

        probs = torch.cat([pos, neg], dim=0)
        labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)], dim=0)

        loss = loss_fun(probs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(train_loader)


@torch.no_grad()
def val_epoch(model: Union[nn.Module, Classifier],
        val_loader: DataLoader,
        loss_fn: nn.Module,
        device: str):

    model.eval()
    correct, total, total_loss = 0, 0, 0
    for a, p, n, *_ in tqdm(val_loader, desc="Validation"):
        a = a.to(device)
        p = p.to(device)
        n = n.to(device)
        pos = model.classify(a, p)
        neg = model.classify(a, n)

        total += 2 * pos.shape[0]
        correct += torch.sum(pos > 0.5).item()
        correct += torch.sum(neg < 0.5).item()

        probs = torch.cat([pos, neg], dim=0)
        labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)], dim=0)
        total_loss += loss_fn(probs, labels).item() * 2 * pos.shape[0]

    metrics = {}
    metrics["loss"] = total_loss / total
    metrics["accuracy"] = correct / total

    return metrics


def main(config_path):

    with open(config_path) as f:
        config = json.load(f)

    # training params
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    device = config["device"]
    patience = config["patience"]
    lr = config["lr"]

    # dataset params
    data_root = config.get("data_root", "data/graph_functions")
    val_subset = config.get("val_subset", 0)
    inp_format = config.get("format", "graph_directed")

    # model params
    if inp_format == "tokens":
        model_type = "lstm"
    else:
        model_type = "gnn"

    train_dataset = FunctionsDataset(data_root, "train.txt", "singles", inp_format, cache="data/cache")
    val_dataset = FunctionsDataset(data_root, "val.txt", "singles", inp_format, val_subset, cache="data/cache")
    num_tokens = train_dataset.num_tokens

    train_dataset = n_triplet_dataset(train_dataset, 100000)
    val_dataset = n_triplet_dataset(val_dataset, 10000)

    train_loader = DataLoader(train_dataset, num_workers=16, pin_memory=True, collate_fn=Collater(), sampler=RandomSampler(train_dataset, replacement=True, num_samples=20000), batch_size=batch_size)
    val_loader = DataLoader(val_dataset, num_workers=16, pin_memory=True, collate_fn=Collater(), batch_size=batch_size)

    
    encoder = Classifier(get_model(model_type, num_tokens, **config["model_params"]))
    encoder.to(device)
    # Set optimizers
    optimizer = optim.Adam(encoder.parameters(), lr=lr)

    lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10, verbose=True)
    loss_func = nn.BCELoss()

    best_metrics = {"loss": float('inf')}
    no_improvement_since = 0
    for epoch in range(1, epochs + 1):
        print(f"EPOCH #{epoch} (BEST LOSS = {best_metrics['loss']})")

        train_epoch(encoder, train_loader, loss_func, optimizer, device)
        # LR scheduler step val loss

        metrics = val_epoch(encoder, val_loader, loss_func, device)
        lr_scheduler.step(metrics["loss"])
        if metrics['loss'] < best_metrics['loss']:
            best_metrics = metrics
            no_improvement_since = 0
            torch.save(encoder.state_dict(), config_path.replace(".json", ".pt"))
        else:
            no_improvement_since += 1

        pprint(metrics)

        if no_improvement_since > patience:
            break

    print("BEST LOSS:", best_metrics["loss"])
    with open(config_path.replace(".json", "_metrics.json"), "w") as f:
        json.dump(best_metrics, f)
    return best_metrics


if __name__ == '__main__':
    loss = main("configs/graphs/config.json")
