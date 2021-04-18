from encoders.classifier import Classifier
from pprint import pprint

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from datasets.utils.collate import Collater
from datasets.functions_dataset import FunctionsDataset
from utils.train_utils import get_model



def train_epoch(model: Classifier,
                loss_fun: nn.Module,
                singles_loader: DataLoader,
                optimizer: optim.Optimizer,
                device: str):
    model.train()

    epoch_loss = 0.
    running_loss = 0.

    optimizer.zero_grad()
    pbar = tqdm(singles_loader, desc="Training")
    for s, l, _ in pbar:

        s = s.to(device)
        probs = model.classify(s)

        loss = loss_fun(probs, l.to(device))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()
        running_loss += loss.item()

    return epoch_loss / len(singles_loader)


@torch.no_grad()
def val_epoch(model: Classifier,
              val_loader: DataLoader,
              device: str):

    model.eval()

    y_true, probs = [], []
    for s, l, _ in tqdm(val_loader, desc="Classification"):
        s = s.to(device)
        prob = model.classify(s).cpu().numpy()
        y_true.append(l.numpy())
        probs.append(prob)
    y_true = np.concatenate(y_true)
    probs = np.concatenate(probs)
    preds = np.argmax(probs, 1)

    accuracies = {}
    accuracies["accuracy"] = accuracy_score(preds, y_true)
    accuracies["f1_score"] = f1_score(preds, y_true, average="macro")

    return accuracies


def main():

    # params
    epochs = 100
    device = "cuda:0"
    batch_size = 256
    model_type = "gnn"
    model_params = {
        "hidden_dim": 32,
        "input_dim": 1,
        "num_layers": 5
    }
    lr = 1e-4
    weight_decay = 1e-6
    patience = 5

    # datasets, dataloaders
    num_classes = 20
    train_dataset = FunctionsDataset("data/graph_functions/", "train.txt", "singles", "graph", split_subset=num_classes, cache="data/cache")
    val_dataset = FunctionsDataset("data/graph_functions/", "train.txt", "singles", "graph", split_subset=20, cache="data/cache")
    train_loader = DataLoader(train_dataset, num_workers=6, pin_memory=True, collate_fn=Collater(), batch_size=batch_size)
    val_loader = DataLoader(val_dataset, num_workers=6, pin_memory=True, collate_fn=Collater(), batch_size=batch_size)

    # model, optimizer, loss
    model = get_model(model_type, train_dataset, num_classes, device, **model_params)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.
    no_improvement_since = 0
    for epoch in range(1, epochs + 1):
        print(f"EPOCH #{epoch} (BEST ACC = {best_acc})")

        loss = train_epoch(model, loss_fn, train_loader, optimizer, device)
        print(loss)

        accuracies = val_epoch(model, val_loader, device)

        acc = accuracies["accuracy"]
        if acc > best_acc:
            best_acc = acc
            no_improvement_since = 0
        else:
            no_improvement_since += 1
        pprint(accuracies)

        if no_improvement_since > patience:
            break

    return best_acc


if __name__ == '__main__':
    best_acc = main()
    print("Training finished! Best accuracy:", best_acc)
