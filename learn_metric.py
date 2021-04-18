import argparse
import json
from os import makedirs, path
from shutil import copyfile
from typing import List, Optional, Union
from pprint import pprint

from torch.optim import lr_scheduler
from utils.train_utils import get_model

import numpy as np
import torch
import umap
import umap.plot
from torch import nn, Tensor
from pytorch_metric_learning import distances, losses, miners, samplers
from pytorch_metric_learning.utils.accuracy_calculator import \
    AccuracyCalculator
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import TensorDataset, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from numba import njit
from sklearn.metrics import accuracy_score, f1_score

from datasets.utils.collate import Collater
from datasets.functions_dataset import FunctionsDataset
from encoders.classifier import Classifier
from utils.measure_performance import measure



# TODO: singles dataset, calculate embeddings, mine hard triplets, classification train on hard triplets


@measure.fun
def train_augs(model: Union[nn.Module, Classifier],
               loss_fun: losses.BaseMetricLossFunction,
               miner: miners.BaseMiner,
               device: str,
               train_loader: DataLoader,
               optimizer: optim.Optimizer,
               update_interval: Union[int, float] = 0.5,
               classification: bool = False):
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

        with measure.block("sample_encoding"):
            samples = samples.to(device)
            embeddings = model(samples).cpu()
            del samples

        with measure.block("sample_encoding"):
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

        # embeddings = torch.cat([embeddings, aug_embeddings], dim=0)
        # labels = torch.cat([labels, labels], dim=0)

            indices_tuple = miner(embeddings, labels)

            emb_loss = loss_fun(embeddings, labels, indices_tuple)

            if classification:
                a = embeddings[indices_tuple[0]].to(device)
                p = embeddings[indices_tuple[1]].to(device)
                n = embeddings[indices_tuple[2]].to(device)
                pred_p = model.classify_embs(a, p)
                pred_n = model.classify_embs(a, n)
                cls_loss_fun = nn.BCELoss()
                cls_loss = cls_loss_fun(torch.cat([pred_p, pred_n], dim=0), torch.cat([torch.ones_like(pred_p), torch.zeros_like(pred_n)], dim=0))
                loss = cls_loss + emb_loss
            else:
                loss = emb_loss
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

    return epoch_loss / len(train_loader)


@njit
def is_sorted(a: np.ndarray):
    for i in range(a.size - 1):
         if a[i+1] < a[i] :
               return False
    return True

cache = dict()

@measure.fun
@torch.no_grad()
def val(model: Union[nn.Module, Classifier],
        val_loader: DataLoader,
        accuracy_calculator: AccuracyCalculator,
        device: str,
        classification: bool = False):

    model.eval()
    embeddings = []
    all_labels = []
    for samples, labels, bases in tqdm(val_loader, desc="Validation"):
        all_labels.append(labels.numpy())

        samples = samples.to(device)
        embs = model(samples).cpu().numpy()
        del samples
        embeddings.append(embs)

    labels = np.squeeze(np.concatenate(all_labels, axis=0))
    embeddings = np.concatenate(embeddings, axis=0)
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

    if classification:
        embeddings = torch.tensor(embeddings)
        labels = torch.tensor(labels)

        if "triplets" not in cache:
            cache["triplets"] = samplers.FixedSetOfTriplets(labels, 100000).fixed_set_of_triplets
        # miner = miners.TripletMarginMiner(distance=distances.CosineSimilarity(), type_of_triplets="hard")
        # indices_tuple = miner(embeddings, labels)
        triplets = cache["triplets"]
        a = embeddings[triplets[:, 0]]
        p = embeddings[triplets[:, 1]]
        n = embeddings[triplets[:, 2]]
        cls_dataset = TensorDataset(a, p, n)
        cls_dataloader = DataLoader(cls_dataset, 512, num_workers=10)
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
        accuracies["accuracy"] = accuracy_score(pred, y_true)
        accuracies["f1_score"] = f1_score(pred, y_true)

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

    runs_dir = "runs"
    current_run_out_dir = path.join(runs_dir, config_path.replace(".json", ""))
    if path.exists(current_run_out_dir):
        from shutil import rmtree
        rmtree(current_run_out_dir)
    log_dir = path.join(current_run_out_dir, "logs")
    makedirs(log_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir)
    models_dir = path.join(current_run_out_dir, "models")
    makedirs(models_dir, exist_ok=True)

    tb_writer.add_hparams(flatten_config(params), {})

    epochs = params["epochs"]
    device = params["device"]
    patience = params["patience"]

    classification = params["classification"]

    train_dataset = FunctionsDataset(**params['dataset'], **params['train_dataset'])
    concat_dataset = ConcatDataset([train_dataset for i in range(params.get("multiplier", 1))])

    val_dataset = FunctionsDataset(**params['dataset'], **params['val_dataset'])

    train_loader = DataLoader(concat_dataset, num_workers=6, pin_memory=True, collate_fn=Collater(), shuffle=True, batch_size=params['batch_size'])
    val_loader = DataLoader(val_dataset, num_workers=6, pin_memory=True, collate_fn=Collater(), batch_size=params['batch_size'])


    encoder = get_model(params["encoder_type"], train_dataset, device=device, **params["encoder"])

    # Set optimizers
    optimizer = optim.Adam(encoder.parameters(), **params["optimizer"])

    # distance = distances.LpDistance()
    distance = distances.CosineSimilarity()
    # TODO?: Try other losses, eg. ContrastiveLoss
    loss_func = losses.TripletMarginLoss(distance=distance)

    miner = miners.TripletMarginMiner(type_of_triplets="all", distance=distance)

    accuracy_calculator = AccuracyCalculator(include=("mean_average_precision_at_r",))


    lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5, verbose=True)
    best_acc = 0.
    no_improvement_since = 0
    best_weights_path = ""
    for epoch in range(1, epochs + 1):
        print(f"EPOCH #{epoch} (BEST ACC = {best_acc})")

        loss = train_augs(encoder, loss_func, miner, device, train_loader, optimizer, classification=classification, **params['training'])
        lr_scheduler.step(loss)
        tb_writer.add_scalar("Loss/Training", loss, epoch)

        accuracies, embs, labels = val(encoder, val_loader, accuracy_calculator, device, classification=classification, **params["val"])

        accuracies["loss"] = loss
        acc = accuracies['mean_average_precision_at_r']
        if classification:
            acc = accuracies["accuracy"]
        if acc > best_acc:
            best_acc = acc
            no_improvement_since = 0
            weights_filename = f"model_{epoch}.pt"
            save_model(encoder, weights_filename, models_dir)
            best_weights_path = path.join(models_dir, weights_filename)
        else:
            no_improvement_since += 1

        tb_writer.add_scalar("MAP@R/Val", accuracies['mean_average_precision_at_r'], epoch)
        if classification:
            tb_writer.add_scalar("F1/Val", accuracies['f1_score'], epoch)
            tb_writer.add_scalar("Accuracy/Val", accuracies['accuracy'], epoch)

        ax = embeddings_visualization(embs, labels, **params["visualization"])
        tb_writer.add_figure("Embeddings", ax.figure, epoch)

        pprint(accuracies)

        if no_improvement_since > patience:
            break

    pprint(measure.summary())

    copyfile(best_weights_path, path.join(models_dir, "best.pt"))
    return best_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training script for R source code plagiarism detection models')
    parser.add_argument('--config_path', '-c', type=str, default='config.json',
        help='Path to config file with parameters, look at config.json')
    args = parser.parse_args()

    best_acc = main(args.config_path)
    print("Training finished! Best accuracy:", best_acc)
