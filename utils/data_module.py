from random import sample
from typing import List, Literal, Optional

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from pytorch_metric_learning import samplers

from datasets.functions_dataset import FunctionsDataset
from datasets.utils.collate import Collater
from utils.train_utils import create_triplet_dataset


class PlagiarismFunctions(LightningDataModule):
    def __init__(self, classify: bool, train_functions: List[str], test_functions: List[str], root_dir: str,
                 format: Literal["graph", "tokens", "letters"], cache: Optional[str] = None, batch_size: int = 128):
        super().__init__()
        self.classify = classify
        self.root_dir = root_dir
        self.format = format
        self.cache = cache
        self.val_functions = sample(train_functions, 100)
        self.train_functions = list(set(train_functions) - set(self.val_functions))
        self.test_functions = test_functions
        self.batch_size = batch_size

    @property
    def num_features(self) -> int:
        return self._num_tokens

    def setup(self, stage: Optional[str] = None):
        print(len(self.train_functions))
        print(len(self.test_functions))
        # train_dataset = FunctionsDataset(self.root_dir, self.train_functions, "singles", self.format, cache=self.cache)
        # self._num_tokens = train_dataset.num_tokens
        # train_triplets = torch.Tensor(samplers.FixedSetOfTriplets(train_dataset.labels, 5 * (10 ** 4)).fixed_set_of_triplets).long()
        # self.train_dataset = create_triplet_dataset(train_dataset, train_triplets)
        if self.classify:
            train_dataset = FunctionsDataset(self.root_dir, self.train_functions, "singles", self.format, cache=self.cache)
            self._num_tokens = train_dataset.num_tokens
            train_triplets = torch.Tensor(samplers.FixedSetOfTriplets(train_dataset.labels, 5 * (10 ** 4)).fixed_set_of_triplets).long()
            self.train_dataset = create_triplet_dataset(train_dataset, train_triplets)
        else:
            self.train_dataset = FunctionsDataset(self.root_dir, self.train_functions, "pairs", self.format, cache=self.cache, multiplier=10)
            self._num_tokens = self.train_dataset.num_tokens

        val_dataset = FunctionsDataset(self.root_dir, self.val_functions, "singles", self.format, cache=self.cache)
        val_triplets = torch.Tensor(samplers.FixedSetOfTriplets(val_dataset.labels, 10**4).fixed_set_of_triplets).long()
        self.val_dataset = create_triplet_dataset(val_dataset, val_triplets)

        if len(self.test_functions) > 0:
            test_dataset = FunctionsDataset(self.root_dir, self.test_functions, "singles", self.format, cache=self.cache)
            test_triplets = torch.Tensor(samplers.FixedSetOfTriplets(test_dataset.labels, 2 * (10**4)).fixed_set_of_triplets).long()
            self.test_dataset = create_triplet_dataset(test_dataset, test_triplets)
        else:
            self.test_dataset = self.val_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          pin_memory=True, num_workers=20, collate_fn=Collater())

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          pin_memory=True, num_workers=20, collate_fn=Collater())

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          pin_memory=True, num_workers=20, collate_fn=Collater())
