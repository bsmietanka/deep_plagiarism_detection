from os import path, getcwd
import pickle
from typing import List, Tuple

from torch.utils import data
from torch import Tensor
import torch

from datasets.utils.r_tokenizer import parse, tokenize, tokens2idxs, chars2idxs, num_tokens, num_chars

__location__ = path.realpath(
    path.join(getcwd(), path.dirname(__file__)))

class NLPDataset(data.Dataset):

    def __init__(self, root_dir: str, tokens=True):
        self.root_path = root_dir
        self.tokens = tokens
        self.vocab_size = num_tokens if self.tokens else num_chars
        pickle_path = path.join(self.root_path, "plagiarism_lists.pickle")

        if not path.exists(pickle_path):
            raise ValueError("Dataset pickle file does not exist")
            # TODO?: create dataset pickle here

        # with open(path.join(__location__, 'errors.txt'), 'r') as f:
        #     self.skip_files = set(f.read().split())

        with open(pickle_path, "rb") as f:
            dataset = pickle.load(f)

        self.labels = []
        self.paths = []
        for i, function_plagiarism in enumerate(dataset):
            for function_path in function_plagiarism:
                # NOTE: try without skip files, but maybe it would be necessary
                self.labels.append(i)
                self.paths.append(path.join(self.root_path, function_path))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index) -> Tuple[Tensor, int]:
        target: int = self.labels[index]
        fun: Tensor = self._tokenize(open(self.paths[index], 'r').read())
        return fun, target

    def _tokenize(self, code: str) -> torch.LongTensor:
        if self.tokens:
            fun_tokens: List[str] = tokenize(code)
            fun = torch.LongTensor(tokens2idxs(fun_tokens))
        else:
            fun_code: str = parse(code)
            fun = torch.LongTensor(chars2idxs(fun_code))
        return fun
