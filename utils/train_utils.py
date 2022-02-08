from typing import Optional, Tuple
from functools import lru_cache
from pytorch_metric_learning import samplers

import torch
from torch import Tensor
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from datasets.functions_dataset import FunctionsDataset, RepresentationType
from encoders import LSTMEncoder, TransformerEncoder, GraphEncoder


def get_model(model_type: str, device: str = "cpu", weights: Optional[str] = None, **kwargs):
    model_type = model_type.lower()
    if model_type == "lstm":
        encoder = LSTMEncoder(**kwargs)
    elif model_type == "transformer":
        encoder = TransformerEncoder(**kwargs)
    elif model_type == "gnn":
        encoder = GraphEncoder(**kwargs)
    else:
        raise ValueError("Unsupported encoder type")

    if weights is not None:
        print("Loading:", weights)
        encoder.load_state_dict(torch.load(weights))

    encoder.to(device)
    return encoder



@lru_cache
def n_triplet_dataset(dataset: FunctionsDataset, n_triplets: int):
    triplets = torch.tensor(samplers.FixedSetOfTriplets(dataset.labels, n_triplets).fixed_set_of_triplets)
    return create_triplet_dataset(dataset, triplets)


def create_pairs_dataset(dataset: FunctionsDataset, pairs: Tensor):

    class PairDataset(Dataset):

        def __init__(self, original_dataset: FunctionsDataset, pairs: Tensor):
            assert original_dataset.singles
            self.original_dataset = original_dataset
            self.pairs = pairs

        def __len__(self) -> int:
            return len(self.pairs)

        def __getitem__(self, index) -> Tuple[RepresentationType, RepresentationType, int]:
            pair = self.pairs[index, :].flatten()
            assert pair.shape == (2,)

            s1, l1, _ = self.original_dataset[pair[0]]
            s2, l2, _ = self.original_dataset[pair[1]]

            return s1, s2, 1 if l1 == l2 else 0

    return PairDataset(dataset, pairs)


def create_triplet_dataset(dataset: FunctionsDataset, triplets: Tensor):

    class TripletDataset(Dataset):

        def __init__(self, original_dataset: FunctionsDataset, triplets: Tensor):
            assert original_dataset.singles
            self.original_dataset = original_dataset
            self.triplets = triplets

        def __len__(self) -> int:
            return len(self.triplets)

        def __getitem__(self, index) -> Tuple[RepresentationType, RepresentationType, RepresentationType]:
            triplet = self.triplets[index, :].flatten()
            assert triplet.shape == (3,)

            a, la, pa = self.original_dataset[triplet[0].item()]
            p, lp, pp = self.original_dataset[triplet[1].item()]
            n, ln, pn = self.original_dataset[triplet[2].item()]

            return a, p, n, la, lp, ln

    return TripletDataset(dataset, triplets)


def pair_tensor_from_pairs(pairs_indices: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
    a1, p, a2, n = pairs_indices
    pairs = torch.stack([torch.cat([a1, a2], dim=0), torch.cat([p, n], dim=0)], dim=1)
    assert len(pairs.shape) == 2 and pairs.shape[1] == 2
    return pairs


def pair_tensor_from_triplets(triplet_indices: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
    a, p, n = triplet_indices
    pairs = torch.stack([torch.cat([a, a], dim=0), torch.cat([p, n], dim=0)], dim=1)
    assert len(pairs.shape) == 2 and pairs.shape[1] == 2
    return pairs


def triplet_tensor_from_triplets(triplet_indices: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
    a, p, n = triplet_indices
    triplets = torch.stack([a, p, n], dim=1)
    assert len(triplets.shape) == 2 and triplets.shape[1] == 3
    return triplets


@torch.no_grad()
def get_embeddings(model: nn.Module, dataloader: DataLoader, device: str, iters: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor]:
    model.eval()
    embeddings, labels, indices = [], [], []
    for i, (samples, ls, index) in enumerate(tqdm(dataloader, desc="Embedding", total=iters)):
        samples = samples.to(device)
        embs = model(samples)
        embeddings.append(embs.cpu())
        labels.append(ls)
        indices.append(index)
        if iters is not None and i >= iters:
            break

    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)
    indices = torch.cat(indices, dim=0)
    return embeddings, labels, indices
