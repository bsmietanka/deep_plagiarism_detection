import argparse
from functools import partial
from os import path
from pathlib import Path
import time
from typing import Callable, List, Literal, Tuple
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from datasets.functions_dataset import FunctionsDataset
from datasets.utils.collate import Collater

from similarity_measures.gst_similarity import GSTSimilarity
from similarity_measures.wl_similarity import WL
from utils.lightning_module import PlagiarismModel


# balanced = ""
balanced = "_unbalanced"


class PairDataset(Dataset):
    def __init__(self, data_root: str, pairs: List[Tuple[str, str]], format: Literal["graph", "tokens"], tensors: bool = True):
        self.data_root = Path(data_root)
        self.pairs = pairs
        self.format = format
        self.tensors = tensors
        self.cache_dir = "data_cache/"

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index):
        p1, p2 = self.pairs[index]
        if self.format == "graph":
            f1 = FunctionsDataset.parse_graph(str(self.data_root / p1), return_tensor=self.tensors)
            f2 = FunctionsDataset.parse_graph(str(self.data_root / p2), return_tensor=self.tensors)
        else:
            f1 = FunctionsDataset.tokenize(str(self.data_root / p1), return_tensor=self.tensors, cache_path=self.cache_dir + str(self.data_root / p1))
            f2 = FunctionsDataset.tokenize(str(self.data_root / p2), return_tensor=self.tensors, cache_path=self.cache_dir + str(self.data_root / p1))
        return f1, f2

@torch.no_grad()
def compute_similarities(similarity_fun: Callable, pair_loader: DataLoader, cuda: bool = False):
    inference_total_time = []

    similarities = []
    for f1, f2 in pair_loader:
        if cuda:
            f1 = f1.cuda()
            f2 = f2.cuda()

        start = time.perf_counter()
        # breakpoint()
        sim = similarity_fun(f1, f2)
        t = time.perf_counter() - start
        inference_total_time.append(t)

        if isinstance(sim, torch.Tensor):
            sim = sim.cpu().flatten().tolist()
        elif isinstance(sim, np.ndarray):
            sim = sim.view(-1).to_list()
        elif isinstance(sim, float):
            sim = [sim]
        similarities.extend(sim)

    return similarities, inference_total_time



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", choices=["gst", "wlk", "gnn", "lstm"])
    parser.add_argument("-w", type=str, default=None)

    args = parser.parse_args()
    if args.m in ["gst", "lstm"]:
        format = "tokens"
        datasets = sorted(list(Path("data/text/").glob(f"**/!pairs{balanced}.csv")))
    else:
        format = "graph"
        datasets = sorted(list(Path("data/graphs/").glob(f"**/!pairs{balanced}.csv")))


    cuda = False
    tensors = True
    if args.m == "gst":
        similarity_fun = GSTSimilarity(13)
        tensors = False
    elif args.m in ["gnn", "lstm"]:
        model = PlagiarismModel.load_from_checkpoint(args.w)
        model.eval().cuda()
        cuda = True
        similarity_fun = partial(model.similarity)
    else:
        similarity_fun = WL(3)

    total_inf_times = []
    for dataset in tqdm(datasets):
        similarities_path: Path = dataset.parent / f"!{'_'.join([args.m, path.basename(args.w)] if args.w is not None else [args.m])}{balanced}.csv"
        if similarities_path.exists():
            continue
        # if str(dataset) == "data/text/alpha=1.5/n=200/p=0.25/r=0.3/10/!pairs.csv":
        #     continue
        pairs_df = pd.read_csv(str(dataset), sep=",", index_col=False, names=["f1","f2","clone"])
        pairs = pairs_df[["f1", "f2"]].to_dict("split")["data"]
        pairs = [tuple(f1f2) for f1f2 in pairs]
        pair_dataset = PairDataset(str(dataset.parent), pairs, format, tensors)
        # if args.m != "gst":
        if args.m == "gst":
            batch_size = 250
        elif args.m == "wlk":
            batch_size = 250
        elif args.m == "gnn":
            batch_size = 250
        else:
            batch_size = 250
        pair_loader = DataLoader(pair_dataset, batch_size, False, pin_memory=True, num_workers=10, collate_fn=Collater())
        similarities, inf_times = compute_similarities(similarity_fun, pair_loader, cuda)

        assert len(similarities) == len(pair_dataset)

        total_inf_times += inf_times
        with similarities_path.open("w") as f:
            f.write("\n".join(map(str, similarities)))

    with open("performance_results.txt", "a") as f:
        f.write(f"{args.m},{args.w},{np.sum(total_inf_times)}\n")
