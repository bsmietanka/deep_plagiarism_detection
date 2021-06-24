from typing import Tuple, Union

from torch import Tensor, LongTensor
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from torch.nn.utils.rnn import pack_sequence, PackedSequence

TensorType = Union[Tensor, Data, PackedSequence]
RepresentationType = Union[TensorType, Tuple[TensorType, ...]]

class Collater:

    def collate(self, batch) -> RepresentationType:
        elem = batch[0]
        if isinstance(elem, Data):
            return Batch.from_data_list(batch)
        elif isinstance(elem, LongTensor):
            return pack_sequence(batch, enforce_sorted=False)
        elif isinstance(elem, tuple):
            tuple_of_lists = tuple(map(list, zip(*batch)))
            return tuple(self.collate(batch_item) for batch_item in tuple_of_lists)
        return default_collate(batch)

    def __call__(self, batch) -> RepresentationType:
        return self.collate(batch)
