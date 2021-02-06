import argparse
import json
from datetime import datetime
from os import makedirs, path
from typing import List, Optional, Union
from pprint import pprint

import numpy as np
import torch
import umap
import umap.plot
from torch import nn
from torch import Tensor
from pytorch_metric_learning import distances, losses, miners
from pytorch_metric_learning.utils.accuracy_calculator import \
    AccuracyCalculator
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from numba import njit

from datasets.utils.collate import Collater
from datasets.functions_dataset import FunctionsDataset
from encoders import GraphEncoder, LSTMEncoder, TransformerEncoder
from utils.measure_performance import measure


runs_dir = "runs"
start_time = datetime.now()
formatted_start_time = start_time.strftime("%y-%m-%d %H:%M:%S")
current_run_out_dir = path.join(runs_dir, formatted_start_time)
log_dir = path.join(current_run_out_dir, "logs")
makedirs(log_dir, exist_ok=True)
tb_writer = SummaryWriter(log_dir)
models_dir = path.join(current_run_out_dir, "models")
makedirs(models_dir, exist_ok=True)


@measure.fun
def train_augs(model: nn.Module,
               loss_fun: losses.BaseMetricLossFunction,
               miner: miners.BaseMiner,
               device: str,
               train_loader: DataLoader,
               optimizer: optim.Optimizer,
               update_interval: Union[int, float] = 0.5):
    model.train()

    epoch_loss = 0.
    running_loss = 0.
    mined_triplets = 0

    accumulation_steps = 1
    if train_loader.batch_size < 40:
        accumulation_steps = np.ceil(40 / train_loader.batch_size)

    if isinstance(update_interval, float):
        update_interval = round(update_interval * len(train_loader))

    n_accumulations = update_interval // accumulation_steps
    if n_accumulations < 1:
        n_accumulations = 1
    update_interval = n_accumulations * accumulation_steps

    accumulated_embeddings: List[Tensor] = []
    accumulated_labels: List[Tensor] = []
    optimizer.zero_grad()
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (samples, aug_samples, labels) in enumerate(pbar, 1):

        with measure.block("sample1_encoding"):
            samples = samples.to(device)
            embeddings = model(samples).cpu()
            del samples

        with measure.block("sample2_encoding"):
            aug_samples = aug_samples.to(device)
            aug_embeddings = model(aug_samples).cpu()
            del aug_samples

        accumulated_embeddings.extend([embeddings, aug_embeddings])
        accumulated_labels.extend([labels, labels])

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            embeddings = torch.cat(accumulated_embeddings, dim=0)
            labels = torch.cat(accumulated_labels, dim=0)
            accumulated_embeddings.clear()
            accumulated_labels.clear()
            indices_tuple = miner(embeddings, labels)
            loss = loss_fun(embeddings, labels, indices_tuple)
            del embeddings
            del labels
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            running_loss += loss.item()
            mined_triplets += miner.num_triplets

        if batch_idx % update_interval == 0:
            running_loss /= n_accumulations
            pbar.set_postfix_str(f"Avg loss = {running_loss}, # mined triplets = {mined_triplets}")
            running_loss = 0.

    return epoch_loss


@njit
def is_sorted(a: np.ndarray):
    for i in range(a.size - 1):
         if a[i+1] < a[i] :
               return False
    return True


@measure.fun
@torch.no_grad()
def val(model: nn.Module,
             val_loader: DataLoader,
             accuracy_calculator: AccuracyCalculator,
             device: str):
    model.eval()
    embeddings = []
    all_labels = []
    for samples, labels in tqdm(val_loader, desc="Validation"):
        all_labels.append(labels.numpy())

        samples = samples.to(device)
        embs = model(samples).cpu().numpy()
        del samples
        embeddings.append(embs)

    labels = np.squeeze(np.concatenate(all_labels, axis=0))
    embeddings = np.concatenate(embeddings, axis=0)
    # TODO: reference embeddings should be 1 from each category
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
    return accuracies, embeddings, labels


@measure.fun
def embeddings_visualization(embeddings: np.ndarray,
                             labels: np.ndarray,
                             subset: Optional[Union[int, float]] = None):
    half_idx = embeddings.shape[0] // 2
    if isinstance(subset, int):
        embeddings = np.concatenate([embeddings[:subset], embeddings[half_idx:half_idx + subset]], axis=0)
        labels = np.concatenate([labels[:subset], labels[half_idx:half_idx + subset]], axis=0)
    elif isinstance(subset, float):
        subset = round(half_idx * subset)
        embeddings = np.concatenate([embeddings[:subset], embeddings[half_idx:half_idx + subset]], axis=0)
        labels = np.concatenate([labels[:subset], labels[half_idx:half_idx + subset]], axis=0)

    mapper = umap.UMAP(metric="cosine").fit(embeddings)
    return umap.plot.points(mapper, labels=labels, theme='fire', height=1200, width=1200)


def save_model(model: nn.Module, name: str, out_dir: str):
    torch.save(model.state_dict(), path.join(out_dir, name))


def flatten_params_dict(params: dict) -> dict:
    out_dict = {}
    for k, v in params.items():
        if isinstance(v, (float, int, str, bool, type(None))):
            out_dict[k] = v
        elif isinstance(v, list):
            out_dict[k] = str(v)
        elif isinstance(v, dict):
            partial = flatten_params_dict(v)
            for kp, vp in partial.items():
                out_dict[f"{k}/{kp}"] = vp
        else:
            raise ValueError("Not supported type for flattening the params dict: ", type(v))
    return out_dict


def flatten_config(config: dict) -> dict:
    out = {}
    for k, v in flatten_params_dict(config).items():
        out[f"hparams/{k}"] = v
    return out


def main(config_path):

    with open(config_path, 'r') as f:
        params = json.load(f)

    tb_writer.add_hparams(flatten_config(params), {})

    epochs = params["epochs"]
    device = params["device"]

    train_dataset = FunctionsDataset(**params['dataset'], **params['train_dataset'])

    val_dataset = FunctionsDataset(**params['dataset'], **params['val_dataset'])

    train_loader = DataLoader(train_dataset, num_workers=6, collate_fn=Collater(), shuffle=True, batch_size=params['batch_size'])
    val_loader = DataLoader(val_dataset, num_workers=6, collate_fn=Collater(), batch_size=params['batch_size'])


    encoder_type = params["encoder_type"].lower()
    if encoder_type == "lstm":
        assert not train_dataset.graph
        encoder = LSTMEncoder(**params['encoder'], vocab_size=train_dataset.num_tokens)
    elif encoder_type == "transformer":
        assert not train_dataset.graph
        encoder = TransformerEncoder(**params['encoder'], vocab_size=train_dataset.num_tokens)
    elif encoder_type == "gnn":
        assert train_dataset.graph
        encoder = GraphEncoder(**params['encoder'], input_dim=train_dataset.num_tokens)
    else:
        raise ValueError("Unsupported encoder type")

    encoder.to(device)

    # Set optimizers
    optimizer = optim.Adam(encoder.parameters(), **params["optimizer"])

    distance = distances.CosineSimilarity()
    # TODO?: Try other losses, eg. ContrastiveLoss
    loss_func = losses.TripletMarginLoss(distance=distance)

    # TODO?: Try other miners, eg. TripletMarginMiner
    miner = miners.TripletMarginMiner(type_of_triplets="semihard", distance=distance)
    # miner = miners.TripletMarginMiner(distance=distance, type_of_triplets="semihard")
    accuracy_calculator = AccuracyCalculator(include=("precision_at_1", "mean_average_precision_at_r"), k = 1)


    best_acc = 0.
    no_improvement_since = 0
    for epoch in range(1, epochs + 1):
        print(f"EPOCH #{epoch}")

        loss = train_augs(encoder, loss_func, miner, device, train_loader, optimizer, **params['training'])
        tb_writer.add_scalar("Loss/Training", loss, epoch)

        accuracies, embs, labels = val(encoder, val_loader, accuracy_calculator, device, **params["val"])

        acc = accuracies['precision_at_1']
        if best_acc < acc:
            best_acc = acc
            no_improvement_since = 0
        else:
            no_improvement_since += 1
        tb_writer.add_scalar("Accuracy/Val", accuracies['precision_at_1'], epoch)

        ax = embeddings_visualization(embs, labels, **params["visualization"])
        tb_writer.add_figure("Embeddings", ax.figure, epoch)

        pprint(measure.summary())

        save_model(encoder, f"model_{epoch}.pt", models_dir)

        if no_improvement_since > 5:
            break

    print("Training finished! Best accuracy:", best_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training script for R source code plagiarism detection models')
    parser.add_argument('--config_path', '-c', type=str, default='config.json',
        help='Path to config file with parameters, look at config.json')
    args = parser.parse_args()

    main(args.config_path)
