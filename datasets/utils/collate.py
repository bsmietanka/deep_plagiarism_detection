from typing import List, Sequence, Union

import torch
from torch import Tensor
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, PackedSequence

# from datasets.utils.graph_attrs import node_attrs

TensorType = Union[Tensor, Data, PackedSequence]
RepresentationType = Union[TensorType, Sequence[TensorType]]

class GraphCollater:

    def collate(self, batch) -> RepresentationType:
        elem = batch[0]
        if isinstance(elem, Data):
            return Batch.from_data_list(batch)
        elif isinstance(elem, tuple):
            tuple_of_lists = tuple(map(list, zip(*batch)))
            return tuple(self.collate(batch_item) for batch_item in tuple_of_lists)
        return default_collate(batch)

    def __call__(self, batch) -> RepresentationType:
        return self.collate(batch)

class NLPCollater:

    @staticmethod
    def _pack_sequences(sequences_batch: List[torch.LongTensor]) -> PackedSequence:
        seq_lengths = [len(seq) for seq in sequences_batch]
        padded_seq = pad_sequence(sequences_batch, batch_first=True)
        packed_seq = pack_padded_sequence(padded_seq, seq_lengths,
                                          batch_first=True, enforce_sorted=True)
        return packed_seq

    def collate(self, batch) -> RepresentationType:
        elem = batch[0]
        if isinstance(elem, torch.LongTensor):
            return self._pack_sequences(batch)
        elif isinstance(elem, tuple):
            tuple_of_lists = tuple(map(list, zip(*batch)))
            return tuple(self.collate(batch_item) for batch_item in tuple_of_lists)
        return default_collate(batch)

    def __call__(self, batch) -> RepresentationType:
        return self.collate(batch)

