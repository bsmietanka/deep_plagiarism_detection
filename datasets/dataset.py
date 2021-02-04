from os import path, getcwd, listdir
import os
from typing import List, Tuple, Optional
from itertools import permutations
from collections import defaultdict
from multiprocessing import Pool

import pandas as pd
from torch.utils import data
from torch import Tensor
import torch
from tqdm import tqdm

from datasets.utils.r_tokenizer import parse, tokenize, tokens2idxs, chars2idxs, num_tokens, num_chars

__location__ = path.realpath(
    path.join(getcwd(), path.dirname(__file__)))

# TODO: create triplet loss dataset
class Dataset(data.Dataset):

    def __init__(self, root_dir: str, class_ratio: float = .0, num_jobs: int = 1, tokens=True):
        self.root_path: str = root_dir
        self.class_ratio = class_ratio
        self.tokens = tokens
        self.vocab_size = num_tokens if self.tokens else num_chars
        csv_path = path.join(self.root_path, "full.csv")

        if path.exists(csv_path):
            self.dataset: pd.DataFrame = pd.read_csv(csv_path, index_col=0)
            return

        with open(path.join(__location__, 'errors.txt'), 'r') as f:
            self.skip_files = set([line.strip() for line in f])

        dirs = []
        for dirpath, _, files in os.walk(self.root_path):
            if len(files) > 0:
                dirs.append(dirpath)

        print(self.root_path)
        if len(dirs) == 0:
            raise ValueError("Provided dataset root path doesn't include any directories")

        self.dataset = None
        with Pool(num_jobs) as p:
            for dataset in tqdm(p.imap_unordered(self._prepare_dataset, dirs), total=len(dirs),
                                                 desc="Processing directories"):
                if self.dataset is None:
                    self.dataset = dataset
                else:
                    self.dataset = self.dataset.append(df, ignore_index=True)

        # self.dataset: pd.DataFrame = datasets[0]
        # for df in tqdm(datasets[1:]):
        #     self.dataset = self.dataset.append(df, ignore_index=True)

        self.dataset.sample(frac=1).reset_index(drop=True, inplace=True)
        self.dataset.to_csv(csv_path)


    def _prepare_dataset(self, dir_name: str) -> pd.DataFrame:
        df = self._process_dir(dir_name)
        df = self._enforce_class_ratio(df, self.class_ratio)
        return df

    def _enforce_class_ratio(self, df: pd.DataFrame, class_ratio: float = .0) -> pd.DataFrame:
        orig_ratio = sum(df['plagiarism']) / len(df)

        if class_ratio > orig_ratio:
            plag_mask = df['plagiarism'] == 1
            plagiarism: pd.DataFrame = df[plag_mask]
            non_plagiarism: pd.DataFrame = df[~plag_mask]

            non_plag_num_to_choose: int = int(len(plagiarism) * (1 - class_ratio) / class_ratio)
            non_plagiarism = non_plagiarism.sample(n=non_plag_num_to_choose)

            df: pd.DataFrame = plagiarism.append(non_plagiarism, ignore_index=True)
            df = df.sample(frac=1).reset_index(drop=True)

        return df

    def _process_dir(self, dataset_dir: str) -> pd.DataFrame:
        if path.exists(path.join(dataset_dir, 'full.csv')):
            dataset = pd.read_csv(path.join(dataset_dir, 'full.csv'), index_col=0)
        else:
            csv_name = "!benchmark.csv"
            csv_path = path.join(dataset_dir, csv_name)
            df: pd.DataFrame = pd.read_csv(csv_path, sep=";", index_col=0)
            df['similarity'] = pd.to_numeric(
                df['similarity'].str.replace(",", "."))
            df['path1'] = f"{dataset_dir}/" + df['functionName1']
            df['path2'] = f"{dataset_dir}/" + df['functionName2']

            dataset_dict = defaultdict(list)

            files = [path.join(dataset_dir, fname)
                    for fname in listdir(dataset_dir) if fname.endswith(".R")]
            files = [*filter(lambda p: path.abspath(p) not in self.skip_files, files)]

            for fun1, fun2 in permutations(files, 2):
                dataset_dict['fun1'].append(fun1)
                dataset_dict['fun2'].append(fun2)
                filter_res = df[(df['path1'] == fun1) & (df['path2'] == fun2)]
                if len(filter_res) >= 1:
                    dataset_dict['plagiarism'].append(1)
                    # print(filter_res['similarity'].head(1))
                    dataset_dict['similarity'].append(filter_res['similarity'].values[0])
                elif len(filter_res) == 0:
                    dataset_dict['plagiarism'].append(0)
                    dataset_dict['similarity'].append(None)
                else: # NOTE: theoretically there should be at most 1 row in filter_res
                    # print(filter_res)
                    raise RuntimeError("Dataset .csv file ill-defined")

            dataset: pd.DataFrame = pd.DataFrame(dataset_dict)
            dataset.to_csv(path.join(dataset_dir, 'full.csv'))

        return dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> Tuple[Tensor, Tensor, float]:
        row = self.dataset.iloc[index, ]
        target: float = row['plagiarism']
        similarity: Optional[float] = row['similarity']
        fun1: Tensor = self._tokenize(open(row['fun1'], 'r').read())
        fun2: Tensor = self._tokenize(open(row['fun2'], 'r').read())
        return fun1, fun2, target, similarity

    def _tokenize(self, code: str) -> torch.LongTensor:
        if self.tokens:
            fun: List[str] = tokenize(code)
            fun = torch.LongTensor(tokens2idxs(fun))
        else:
            fun: str = parse(code)
            fun = torch.LongTensor(chars2idxs(fun))
        return fun

if __name__ == "__main__":
    d = Dataset("data", class_ratio=0.5, num_jobs=8)
    # d = Dataset("data/alpha=1.0/n=200/p=0.1/r=0.2/2")
    # print(len(d))
    # print(sum(d.dataset['plagiarism']) / len(d))
    # d = Dataset("data/alpha=1.0/n=200/p=0.1/r=0.2/2", class_ratio=0.5)
    # print(len(d))
    # print(sum(d.dataset['plagiarism']) / len(d))
