import json
from pprint import pprint
from typing import Union

import numpy as np
import torch
from numba import njit
from pytorch_metric_learning import (distances, losses, regularizers,
                                     samplers)
from pytorch_metric_learning.utils.accuracy_calculator import \
    AccuracyCalculator
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import RandomSampler
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from datasets.functions_dataset import FunctionsDataset
from datasets.utils.collate import Collater
from encoders.classifier import Classifier
from encoders.graph_encoder import GraphEncoder
from utils.measure_performance import measure
from utils.train_utils import create_triplet_dataset, get_model, n_triplet_dataset

# TODO: singles dataset, calculate embeddings, mine hard triplets, classification train on hard triplets


@measure.fun
def train_augs(model: Union[nn.Module, Classifier],
               loss_fun: losses.BaseMetricLossFunction,
               device: str,
               train_loader: DataLoader,
               optimizer: optim.Optimizer):
    model.train()

    epoch_loss = 0.
    running_loss = 0.

    optimizer.zero_grad()
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (a, p, n, la, lp, ln) in enumerate(pbar, 1):
        optimizer.zero_grad()

        a = a.to(device)
        a_emb= model(a)
        del a
        
        p = p.to(device)
        p_emb = model(p)
        del p

        n = n.to(device)
        n_emb = model(n)
        del n

        embs = torch.cat((a_emb, p_emb, a_emb, n_emb), dim=0)

        indices = None
        labels = torch.cat((la, lp, la, ln), dim=0)

        loss = loss_fun(embs, labels, indices)

        del embs
        del labels
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        running_loss += loss.item()

    epoch_loss /= len(train_loader)
    print(f"Epoch loss = {epoch_loss}")

    return epoch_loss


@njit(cache=True)
def find_first(vec, item):
    """return the index of the first occurence of item in vec"""
    for i, v in enumerate(vec):
        if item == v:
            return i
    return -1


@measure.fun
@torch.no_grad()
def val(model: Union[nn.Module, Classifier],
        val_loader: DataLoader,
        accuracy_calculator: AccuracyCalculator,
        device: str,
        loss_fn: losses.BaseMetricLossFunction):

    model.eval()
    embeddings = []
    all_labels = []
    indices = [[], [], [], []]
    start = 0
    for a, p, n, la, lp, ln in tqdm(val_loader, desc="Validation"):
        la, lp, ln = la.numpy(), lp.numpy(), ln.numpy()
        a = a.to(device)
        a_emb = model(a).cpu().numpy()
        del a
        
        p = p.to(device)
        p_emb = model(p).cpu().numpy()
        del p

        n = n.to(device)
        n_emb = model(n).cpu().numpy()
        del n

        embeddings.extend([a_emb, p_emb, a_emb, n_emb])
        for i, e in enumerate([a_emb, p_emb, a_emb, n_emb]):
            end = start + e.shape[0]
            indices[i].append(np.arange(start, end))
            start = end
        all_labels.extend([la, lp, la, ln])


    for i in range(len(indices)):
        indices[i] = torch.tensor(np.concatenate(indices[i], axis=0))

    labels = np.squeeze(np.concatenate(all_labels, axis=0))
    embeddings = np.concatenate(embeddings, axis=0)
    labels_vals = np.unique(labels)
    idxs = []
    for v in labels_vals:
        idxs.append(find_first(labels, v))
    mask = np.ones(len(labels)).astype(bool)
    mask[idxs] = False

    accuracies = accuracy_calculator.get_accuracy(embeddings[mask],
                                                  embeddings[~mask],
                                                  labels[mask],
                                                  labels[~mask],
                                                  False)

    accuracies["loss"] = loss_fn(torch.tensor(embeddings), torch.tensor(labels), indices).item()

    return accuracies


def main(config_path: str):

    with open(config_path) as f:
        config = json.load(f)

    # training params
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    device = config["device"]
    patience = config["patience"]
    lr = config["lr"]
    loss_func = config.get("loss", "contrastive")

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

    encoder = Classifier(get_model(model_type, train_dataset.num_tokens, **config["model_params"]))
    encoder.to(device)

    train_dataset = n_triplet_dataset(train_dataset, 100000)
    val_dataset = n_triplet_dataset(val_dataset, 10000)

    train_loader = DataLoader(train_dataset, num_workers=12, pin_memory=True, collate_fn=Collater(), sampler=RandomSampler(train_dataset, replacement=True, num_samples=20000), batch_size=batch_size)
    val_loader = DataLoader(val_dataset, num_workers=12, pin_memory=True, collate_fn=Collater(), batch_size=batch_size)

    # Set optimizers
    optimizer = optim.Adam(encoder.parameters(), lr=lr)

    distance = distances.CosineSimilarity()
    if loss_func == "contrastive":
        loss_func = losses.ContrastiveLoss(pos_margin=1, neg_margin=0, distance=distance, embedding_regularizer=regularizers.LpRegularizer())
    if loss_func == "triplet":
        loss_func = losses.TripletMarginLoss(margin=0.8, distance=distance, embedding_regularizer=regularizers.LpRegularizer())
    accuracy_calculator = AccuracyCalculator(include=("mean_average_precision_at_r",))

    lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5, verbose=True)

    best_metrics = {"loss": float('inf')}
    no_improvement_since = 0
    for epoch in range(1, epochs + 1):
        print(f"EPOCH #{epoch} (BEST LOSS = {best_metrics['loss']})")

        train_augs(encoder, loss_func, device, train_loader, optimizer)

        metrics = val(encoder, val_loader, accuracy_calculator, device, loss_func)
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
    main("configs/tokens/config_num_layers_5.json")
