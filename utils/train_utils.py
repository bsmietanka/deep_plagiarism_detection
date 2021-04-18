from typing import Optional, Tuple

import torch
from torch import Tensor
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from datasets.functions_dataset import FunctionsDataset, RepresentationType
from encoders import Classifier, LSTMEncoder, TransformerEncoder, GraphEncoder


def get_model(model_type: str, dataset: FunctionsDataset, num_classes: int = 1, device: str = "cuda:0", weights: Optional[str] = None, **kwargs):
    model_type = model_type.lower()
    if model_type == "lstm":
        assert not dataset.graph
        encoder = Classifier(LSTMEncoder(**kwargs, vocab_size=dataset.num_tokens), num_classes)
    elif model_type == "transformer":
        assert not dataset.graph
        encoder = Classifier(TransformerEncoder(**kwargs, vocab_size=dataset.num_tokens), num_classes)
    elif model_type == "gnn":
        assert dataset.graph
        encoder = Classifier(GraphEncoder(**kwargs, node_labels=dataset.num_tokens), num_classes)
    else:
        raise ValueError("Unsupported encoder type")

    if weights is not None:
        encoder.load_state_dict(torch.load(weights))

    encoder.to(device)
    return encoder


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

            a, *_ = self.original_dataset[triplet[0]]
            p, *_ = self.original_dataset[triplet[1]]
            n, *_ = self.original_dataset[triplet[2]]

            return a, p, n

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
