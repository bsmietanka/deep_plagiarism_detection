
from encoders.graph_encoder import GraphEncoder
from encoders.lstm_encoder import LSTMEncoder
from typing import Union
from pprint import pprint
from utils.train_utils import create_triplet_dataset


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

    accuracies = {}
    accuracies["loss"] = total_loss / total
    accuracies["accuracy"] = correct / total

    return accuracies


def main():

    batch_size = 128
    epochs = 100
    device = "cuda:0"
    patience = 20
    lr = 1e-3

    train_dataset = FunctionsDataset("data/graph_functions", "train.txt", "singles", "graph")
    num_tokens = train_dataset.num_tokens
    val_dataset = FunctionsDataset("data/graph_functions", "val.txt", "singles", "graph")


    train_triplets = torch.tensor(samplers.FixedSetOfTriplets(train_dataset.labels, 10000 * 50).fixed_set_of_triplets)
    train_dataset = create_triplet_dataset(train_dataset, train_triplets)
    val_triplets = torch.tensor(samplers.FixedSetOfTriplets(val_dataset.labels, 10**4).fixed_set_of_triplets)
    val_dataset = create_triplet_dataset(val_dataset, val_triplets)

    # train_triplets = torch.tensor(samplers.FixedSetOfTriplets(train_dataset.labels, batch_size).fixed_set_of_triplets)
    # train_dataset = create_triplet_dataset(train_dataset, train_triplets)
    # val_dataset = train_dataset

    train_loader = DataLoader(train_dataset, num_workers=16, pin_memory=True, collate_fn=Collater(), sampler=RandomSampler(train_dataset, replacement=True, num_samples=20000), batch_size=batch_size)
    val_loader = DataLoader(val_dataset, num_workers=16, pin_memory=True, collate_fn=Collater(), batch_size=batch_size)

    encoder = Classifier(GraphEncoder(1, 512, node_labels=num_tokens, num_layers=5, node_embeddings=5, train_eps=False))
    encoder.to(device)
    # Set optimizers
    optimizer = optim.Adam(encoder.parameters(), lr=lr)

    lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10, verbose=True)
    loss_func = nn.BCELoss()

    best_acc = float('inf')
    no_improvement_since = 0
    for epoch in range(1, epochs + 1):
        print(f"EPOCH #{epoch} (BEST ACC = {best_acc})")

        loss = train_epoch(encoder, train_loader, loss_func, optimizer, device)
        print("TRAIN LOSS:", loss)
        # LR scheduler step val loss

        accuracies = val_epoch(encoder, val_loader, loss_func, device)
        lr_scheduler.step(accuracies["loss"])
        acc = accuracies['loss']
        if acc < best_acc:
            best_acc = acc
            no_improvement_since = 0
            torch.save(encoder.state_dict(), "exp_graph_best.pt")
        else:
            no_improvement_since += 1

        pprint(accuracies)

        if no_improvement_since > patience:
            break

    print("BEST ACC:", best_acc)


if __name__ == '__main__':
    main()
