from pprint import pprint

import numpy as np
import torch
from torch import nn
from pytorch_metric_learning import distances, losses, miners, samplers
from pytorch_metric_learning.utils.accuracy_calculator import \
    AccuracyCalculator
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import TensorDataset, ConcatDataset
from tqdm import tqdm
from numba import jit
from sklearn.metrics import accuracy_score, f1_score

from encoders import Classifier
from datasets.utils.collate import Collater
from datasets.functions_dataset import FunctionsDataset
from utils.measure_performance import measure
from utils.train_utils import create_pairs_dataset, get_embeddings, get_model, pair_tensor_from_pairs, pair_tensor_from_triplets



def train_epoch(model: Classifier,
                loss_fun: nn.Module,
                pair_loader: DataLoader,
                optimizer: optim.Optimizer,
                device: str):
    model.train()

    epoch_loss = 0.
    running_loss = 0.

    optimizer.zero_grad()
    pbar = tqdm(pair_loader, desc="Training")
    for s1, s2, l in pbar:

        s1, s2 = s1.to(device), s2.to(device)
        probs = model.classify(s1, s2).flatten()

        loss = loss_fun(probs, l.float().to(device))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()
        running_loss += loss.item()

    return epoch_loss / len(pair_loader)


def is_sorted(a: np.ndarray):
    for i in range(a.size - 1):
         if a[i+1] < a[i] :
               return False
    return True

cache = dict()

@torch.no_grad()
def val_epoch(model: Classifier,
              val_loader: DataLoader,
              accuracy_calculator: AccuracyCalculator,
              device: str):

    model.eval()
    
    embeddings, labels, indices = get_embeddings(model, val_loader, device)

    if "triplets" not in cache:
        cache["triplets"] = samplers.FixedSetOfTriplets(labels, 100000).fixed_set_of_triplets

    triplets = cache["triplets"]
    a = embeddings[triplets[:, 0]]
    p = embeddings[triplets[:, 1]]
    n = embeddings[triplets[:, 2]]
    cls_dataset = TensorDataset(a, p, n)
    cls_dataloader = DataLoader(cls_dataset, val_loader.batch_size, num_workers=10)
    prob = []
    y_true = []
    for ba, bp, bn in tqdm(cls_dataloader, desc="Classification"):
        ba = ba.to(device)
        prob_p = model.classify_embs(ba, bp.to(device)).cpu().numpy()
        prob_n = model.classify_embs(ba, bn.to(device)).cpu().numpy()
        y_true.extend([np.ones_like(prob_p), np.zeros_like(prob_n)])
        prob.extend([prob_p, prob_n])
    y_true = np.concatenate(y_true)
    prob = np.concatenate(prob)
    pred = prob > 0.5

    labels = labels.numpy()
    embeddings = embeddings.numpy()

    assert is_sorted(labels), "Validation DataLoader should not shuffle data"
    labels_vals = np.unique(labels)
    labels_first_idxs = np.searchsorted(labels, labels_vals)
    mask = np.ones(len(labels)).astype(bool)
    mask[labels_first_idxs] = False

    accuracies = accuracy_calculator.get_accuracy(embeddings[mask],
                                                  embeddings[~mask],
                                                  labels[mask],
                                                  labels[~mask],
                                                  False)

    accuracies["accuracy"] = accuracy_score(pred, y_true)
    accuracies["f1_score"] = f1_score(pred, y_true)

    return accuracies


def main():

    # params
    epochs = 100
    device = "cuda:0"
    multiplier = 1
    batch_size = 256
    model_type = "gnn"
    model_params = {
        "hidden_dim": 256,
        "num_layers": 4,
        "input_dim": 1,
        "node_embeddings": 4
    }
    lr = 1e-3
    weight_decay = 1e-6
    patience = 5

    # datasets, dataloaders
    train_dataset = FunctionsDataset("data/graph_functions/", "train.txt", "singles", "graph", cache="data/cache")
    concat_dataset = ConcatDataset([train_dataset for i in range(multiplier)])
    val_dataset = FunctionsDataset("data/graph_functions/", "val.txt", "singles", "graph", cache="data/cache")
    train_sampler = samplers.MPerClassSampler(train_dataset.labels, 4, batch_size)
    train_loader = DataLoader(concat_dataset, num_workers=12, pin_memory=True, collate_fn=Collater(), sampler=train_sampler, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, num_workers=12, pin_memory=True, collate_fn=Collater(), batch_size=batch_size)

    # model, optimizer, loss
    model = get_model(model_type, train_dataset, 1, device, **model_params)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCELoss()

    # metric learning stuff
    distance = distances.CosineSimilarity()
    miner = miners.TripletMarginMiner(margin=0.2, distance=distance, type_of_triplets="hard")
    acc_calc = AccuracyCalculator()

    best_acc = 0.
    no_improvement_since = 0
    for epoch in range(1, epochs + 1):
        print(f"EPOCH #{epoch} (BEST ACC = {best_acc})")

        embeddings, labels, indices = get_embeddings(model, train_loader, device, 2)
        print("Mining hard triplets")
        a, p, n = miner(embeddings, labels)
        a = indices[a]
        p = indices[p]
        n = indices[n]

        pairs = pair_tensor_from_triplets((a, p, n))
        pair_dataset = create_pairs_dataset(train_dataset, pairs)
        pair_loader = DataLoader(pair_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, collate_fn=Collater())

        loss = train_epoch(model, loss_fn, pair_loader, optimizer, device)
        print(loss)

        accuracies = val_epoch(model, val_loader, acc_calc, device)

        acc = accuracies["accuracy"]
        if acc > best_acc:
            best_acc = acc
            no_improvement_since = 0
            torch.save(model.state_dict(), "exp_mining.pt")
        else:
            no_improvement_since += 1
        pprint(accuracies)

        if no_improvement_since > patience:
            break

    return best_acc


if __name__ == '__main__':
    best_acc = main()
    print("Training finished! Best accuracy:", best_acc)
