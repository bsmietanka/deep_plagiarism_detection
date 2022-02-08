from os import makedirs, path, listdir
import pickle
from typing import Dict, List, Optional, Tuple, Union, Set
import random
from collections import defaultdict

import torch
from torch import LongTensor, FloatTensor
from torch.utils import data
from tqdm import tqdm
from networkx import Graph
try:
    from torch_geometric.utils.loop import remove_self_loops
    from torch_geometric.data import Data as geoData
    from torch_geometric.utils import to_networkx
except:
    pass
from datasets.utils.r_tokenizer import parse, tokenize, tokens2idxs, chars2idxs, num_tokens, num_chars
from datasets.utils.graph_attrs import node_attrs
from datasets.functions_errors import functions_errors

TokensType = Union[LongTensor, str, List[str]]
GraphType = Union[Graph, geoData]
RepresentationType = Union[TokensType, GraphType]
TripletType = Tuple[RepresentationType, RepresentationType, RepresentationType, int] # Anchor, Positive, Negative, Anchor index
PairType = Tuple[RepresentationType, RepresentationType, int, int] # Sample1, Sample2, Sample1 index, Sample2 index
SingleType = Tuple[RepresentationType, int, int] # Sample, index of base function
DatasetItemType = Union[TripletType, PairType, SingleType]


def load_cache(cache_path: Optional[str], tensor: bool):
    if cache_path is not None:
        if tensor:
            return torch.load(cache_path)
        else:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
    raise Exception

def save_cache(obj, cache_path: str, tensor: bool):
    if cache_path is not None:
        makedirs(path.dirname(cache_path), exist_ok=True)
        if tensor:
            torch.save(obj, cache_path)
        else:
            with open(cache_path, "wb") as f:
                pickle.dump(obj, f)


class FunctionsDataset(data.Dataset):

    def __init__(self,
                 root_dir: str,
                 split: Union[str, List[str]],
                 mode: str = "pairs",
                 format: str = "tokens",
                 split_subset: Union[int, Tuple[int, int]] = 0,
                 multiplier: int = 1,
                 cache: Optional[str] = None,
                 return_tensor: bool = True,
                 only_augs: bool = True,
                 remove_self_loops: bool = False):

        self.return_tensor = return_tensor
        self.remove_self_loops = remove_self_loops
        self.only_augs = only_augs
        self.augs = True
        self.multiplier = multiplier

        self.format = format.lower()
        assert self.format in ["graph", "tokens", "letters", "graph_directed"]

        self.mode = mode
        assert self.mode in ["singles", "pairs", "triplets"]

        self.cache = cache

        if self.tokens:
            self.num_tokens = num_tokens
        elif self.letters:
            self.num_tokens = num_chars
        else:
            self.num_tokens = len(node_attrs)

        required_extension = ".R.txt" if self.graph else ".R"

        # ERRORS
        tokenizing_errors = defaultdict(set)
        for fpath in functions_errors:
            base, src_file = fpath.split("/")
            tokenizing_errors[base].add(src_file)

        self.root_dir: str = path.join(root_dir, "")
        if isinstance(split, str):
            with open(path.join(self.root_dir, split)) as f:
                self.function_bases = list(map(lambda l: l.strip(), f.readlines()))
        else:
            self.function_bases = split

        if isinstance(split_subset, int) and split_subset > 0:
            self.function_bases = self.function_bases[:split_subset]
        elif isinstance(split_subset, tuple) and len(split_subset) == 2:
            self.function_bases = self.function_bases[split_subset[0]:split_subset[1]]

        src_files: Dict[str, List[str]] = {}
        for fun_base in self.function_bases:
            paths = listdir(path.join(self.root_dir, fun_base))
            paths = set(filter(lambda p: p.endswith(required_extension), paths))
            paths = list(paths - tokenizing_errors[fun_base])
            paths = [path.join(self.root_dir, fun_base, fpath) for fpath in paths]
            src_files[fun_base] = paths

        if self.singles:
            self.labels = []
            self.src_files = []
            for base, funs in tqdm(src_files.items()):
                self.labels.extend(([self.function_bases.index(base)] * len(funs)))
                self.src_files.extend(funs)
        else:
            self.src_files = src_files

        self.indexes: Set[int] = set(range(len(self.function_bases)))


    @property
    def graph(self):
        return "graph" in self.format

    @property
    def tokens(self):
        return self.format == "tokens"

    @property
    def letters(self):
        return self.format == "letters"

    @property
    def nlp(self):
        return "graph" not in self.format

    @property
    def directed(self):
        return self.format == "graph_directed"

    @property
    def pairs(self):
        return self.mode == "pairs"

    @property
    def triplets(self):
        return self.mode == "triplets"

    @property
    def singles(self):
        return self.mode == "singles"


    def num_functions(self) -> int:
        return len(self.function_bases)


    def __len__(self) -> int:
        return self.multiplier * len(self.src_files)


    def __getitem__(self, index: int) -> DatasetItemType:
        index = index % len(self.src_files)
        if self.triplets:
            return self._get_triplet(index)
        elif self.pairs:
            return self._get_pair(index)
        else:
            return self._get_single(index)


    def _get_triplet(self, index: int) -> TripletType:
        # anchor
        base = self.function_bases[index]

        base_src_files = self.src_files[base]
        if len(base_src_files) > 1:
            anchor_path, positive_path = random.sample(base_src_files, 2)
        else:
            anchor_path, positive_path = base_src_files[0], base_src_files[0]

        #negative
        negative_path = self._get_negative_path(index)

        # tokenizing
        anchor_tokens = self._parse_src(anchor_path)
        positive_tokens = self._parse_src(positive_path)
        negative_tokens = self._parse_src(negative_path)

        return anchor_tokens, positive_tokens, negative_tokens, index


    # returns only positive samples, similar to unsupervised/semi-supervised learning, sample2 is augmented version of sample1
    def _get_pair(self, index: int) -> PairType:
        base = self.function_bases[index]
        base_src_files = self.src_files[base]
        if self.augs:
            index2 = index
            if len(base_src_files) > 1:
                sample1_path, sample2_path = random.sample(base_src_files, 2)
            else:
                sample1_path, sample2_path = base_src_files[0], base_src_files[0]
        else:
            sample1_path = random.choice(base_src_files)
            base2 = base_src_files
            while base2 == base_src_files:
                base2 = random.choice(self.function_bases)
            index2 = self.function_bases.index(base2)
            sample2_src_files = self.src_files[base2]
            sample2_path = random.choice(sample2_src_files)

        if not self.only_augs:
            self.augs = not self.augs

        sample1 = self._parse_src(sample1_path)
        sample2 = self._parse_src(sample2_path)

        return sample1, sample2, index, index2


    def _get_single(self, index: int) -> SingleType:
        src_file: str = self.src_files[index]
        base = path.dirname(src_file.replace(self.root_dir, ""))

        sample = self._parse_src(src_file)
        return sample, self.function_bases.index(base), index


    def _get_negative_path(self, anchor) -> str:
        all_except_anchor: List[int] = list(self.indexes - {anchor})
        negative_base = self.function_bases[random.choice(all_except_anchor)]
        return random.choice(self.src_files[negative_base])


    def _get_cache_path(self, src_path) -> Optional[str]:
        if self.cache is None:
            return None
        relative_src_path = src_path.replace(self.root_dir, "")
        cache_src_path = path.join(self.cache, self.format.replace("_directed", ""), relative_src_path + ".pt")
        return cache_src_path


    def _parse_src(self, src_path: str) -> RepresentationType:
        cache_path = self._get_cache_path(src_path)
        if cache_path is not None:
            try:
                return torch.load(cache_path)
            except:
                pass # fallback to standard parsing
        if self.graph:
            parsed_data = self.parse_graph(src_path, self.directed, self.remove_self_loops, self.return_tensor)
        else:
            parsed_data = self.tokenize(src_path, not self.tokens, self.return_tensor)
        return parsed_data

    @staticmethod
    def parse_graph(src_path: str, directed: bool = False, remove_loops: bool = True,
                     return_tensor: bool = True, cache_path: str = None) -> GraphType:

        try:
            return load_cache(cache_path, return_tensor)
        except:
            pass
        with open(src_path, 'r') as f:
            nodes = list(map(int, f.readline().strip().split(",")))
            # nodes = [n if n < 30 else n // 10 for n in nodes]
            nodes = list(map(lambda x: node_attrs.index(x), nodes))

            edges = [[], []]
            edge_attrs = []
            for line in f:
                src, dst, tp = map(int, line.strip().split(","))
                edges[0].append(src)
                edges[1].append(dst)
                if not directed:
                    edges[0].append(dst)
                    edges[1].append(src)
                edge_attrs.append(tp)
                edge_attrs.append(tp)

        nodes = LongTensor(nodes).reshape(-1, 1)
        edges = LongTensor(edges)
        edge_attrs = FloatTensor(edge_attrs).reshape(-1, 1)
        graph = geoData(x=nodes, edge_index=edges, edge_attr=edge_attrs)
        if remove_loops:
            graph.edge_index, graph.edge_attr = remove_self_loops(graph.edge_index, graph.edge_attr)
        if not return_tensor:
            graph = to_networkx(graph)
        save_cache(graph, cache_path, return_tensor)
        return graph

    @staticmethod
    def tokenize(src_path: str, chars: bool = False, return_tensor: bool = True, cache_path: str = None) -> TokensType:
        try:
            return load_cache(cache_path, return_tensor)
        except:
            pass
        with open(src_path) as f:
            code = f.read()
        if not chars:
            tokens: List[str] = tokenize(code)
            if not return_tensor:
                return tokens
            idx_tokens: List[int] = tokens2idxs(tokens)
            return LongTensor(idx_tokens).view(-1, 1)
        parsed_code: str = parse(code)
        if not return_tensor:
            save_cache(parsed_code, cache_path, return_tensor)
            return parsed_code
        t = LongTensor(chars2idxs(parsed_code)).view(-1, 1)
        save_cache(t, cache_path, return_tensor)
        return t
