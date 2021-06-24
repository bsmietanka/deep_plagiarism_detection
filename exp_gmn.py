from typing import List, Optional, Union
from pprint import pprint
from utils.train_utils import create_triplet_dataset


import numpy as np
import torch
from torch import nn, Tensor
from pytorch_metric_learning import distances, losses, miners, samplers, regularizers
from pytorch_metric_learning.utils.accuracy_calculator import \
    AccuracyCalculator
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import TensorDataset, ConcatDataset, RandomSampler
from tqdm import tqdm
from numba import njit
from sklearn.metrics import accuracy_score, f1_score

from datasets.utils.collate import Collater
from datasets.functions_dataset import FunctionsDataset
from encoders.classifier import Classifier
from encoders.graph_matching_network import GraphMatchingNetwork
from utils.measure_performance import measure



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
        p = p.to(device)
        a1_emb, p_emb = model(a, p)
        del p

        n = n.to(device)
        a2_emb, n_emb = model(a, n)
        del a, n

        embs = torch.cat((a1_emb, p_emb, a2_emb, n_emb), dim=0)
        # labels = torch.cat(
        #     (
        #         torch.ones((2 * a1_emb.shape[0],)),
        #         torch.zeros((2 * a2_emb.shape[0],))
        #     ),
        #     dim=0)
        # indices = []
        # start = 0
        # for e in [a1_emb, p_emb, a2_emb, n_emb]:
        #     end = start + e.shape[0]
        #     indices.append(torch.arange(start, end))
        #     start = end
        # indices = tuple(indices)

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
        p = p.to(device)
        a1_emb, p_emb = model(a, p)
        a1_emb, p_emb = a1_emb.cpu().numpy(), p_emb.cpu().numpy()
        del p

        n = n.to(device)
        a2_emb, n_emb = model(a, n)
        a2_emb, n_emb = a2_emb.cpu().numpy(), n_emb.cpu().numpy()
        del a, n

        embeddings.extend([a1_emb, p_emb, a2_emb, n_emb])
        for i, e in enumerate([a1_emb, p_emb, a2_emb, n_emb]):
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


def main():

    batch_size = 256
    epochs = 100
    device = "cuda:0"
    patience = 20
    lr = 1e-3

    train_dataset = FunctionsDataset("data/graph_functions", "train.txt", "singles", "graph", cache="data/cache")
    val_dataset = FunctionsDataset("data/graph_functions", "val.txt", "singles", "graph", cache="data/cache")

    encoder = GraphMatchingNetwork(train_dataset.num_tokens, 4, [256, 256, 256, 256])
    encoder.to(device)

    train_triplets = torch.tensor(samplers.FixedSetOfTriplets(train_dataset.labels, 5000 * 50).fixed_set_of_triplets)
    train_dataset = create_triplet_dataset(train_dataset, train_triplets)
    val_triplets = torch.tensor(samplers.FixedSetOfTriplets(val_dataset.labels, 10**4).fixed_set_of_triplets)
    val_dataset = create_triplet_dataset(val_dataset, val_triplets)

    train_loader = DataLoader(train_dataset, num_workers=12, pin_memory=True, collate_fn=Collater(), sampler=RandomSampler(train_dataset, replacement=True, num_samples=5000), batch_size=batch_size)
    val_loader = DataLoader(val_dataset, num_workers=20, pin_memory=True, collate_fn=Collater(), batch_size=batch_size)

    # Set optimizers
    optimizer = optim.Adam(encoder.parameters(), lr=lr)

    # distance = distances.LpDistance()
    distance = distances.CosineSimilarity()
    # TODO?: Try other losses, eg. ContrastiveLoss
    loss_func = losses.ContrastiveLoss(pos_margin=1, neg_margin=0, distance=distance, embedding_regularizer=regularizers.LpRegularizer())
    accuracy_calculator = AccuracyCalculator(include=("mean_average_precision_at_r",))

    lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5, verbose=True)

    best_acc = 0.
    no_improvement_since = 0
    for epoch in range(1, epochs + 1):
        print(f"EPOCH #{epoch} (BEST ACC = {best_acc})")

        loss = train_augs(encoder, loss_func, device, train_loader, optimizer)
        # LR scheduler step val loss

        accuracies = val(encoder, val_loader, accuracy_calculator, device, loss_func)
        lr_scheduler.step(accuracies["loss"])
        acc = accuracies['mean_average_precision_at_r']
        if acc > best_acc:
            best_acc = acc
            no_improvement_since = 0
        else:
            no_improvement_since += 1

        pprint(accuracies)

        if no_improvement_since > patience:
            break

    print("BEST ACC:", best_acc)


if __name__ == '__main__':
    main()
