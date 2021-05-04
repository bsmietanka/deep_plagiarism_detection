from typing import Union

import numpy as np
from torch import LongTensor, no_grad
from torch_geometric.data import Data, Batch
from numba import njit

from utils.train_utils import get_model

SequenceType = Union[LongTensor, Data]


@njit
def cosine_similarity(x1: np.ndarray, x2: np.ndarray) -> float:
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


class DeepSimilarity:

    def __init__(self, *args, **kwargs):
        self.device = kwargs["device"]
        self.encoder = get_model(**kwargs)
        self.encoder.eval()


    def _batch(self, f: SequenceType) -> SequenceType:
        if isinstance(f, Data):
            return Batch.from_data_list([f]).to(self.device)
        return f.unsqueeze(0).to(self.device)


    @no_grad()
    def __call__(self, f1: SequenceType, f2: SequenceType) -> float:
        f1 = self._batch(f1)
        f2 = self._batch(f2)

        emb1 = self.encoder(f1).view(-1).cpu().numpy()
        emb2 = self.encoder(f2).view(-1).cpu().numpy()

        return cosine_similarity(emb1, emb2)
