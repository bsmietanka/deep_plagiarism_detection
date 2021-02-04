from os import path, listdir
from typing import Dict, Iterable, List, Tuple, Union, Set
import random
from itertools import chain
from collections import defaultdict

from torch.utils import data
import torch
from torch_geometric.data import Data as geoData

from datasets.utils.r_tokenizer import parse, tokenize, tokens2idxs, chars2idxs, num_tokens, num_chars
from datasets.utils.graph_attrs import node_attrs
from datasets.functions_errors import functions_errors

RepresentationType = Union[torch.LongTensor, geoData]
TripletType = Tuple[RepresentationType, RepresentationType, RepresentationType, int] # Anchor, Positive, Negative
PairType = Tuple[RepresentationType, RepresentationType, int] # Sample1, Sample2, Plagiarism?
SingleType = Tuple[RepresentationType, int]
DatasetItemType = Union[TripletType, PairType, SingleType] # Triplet or Pair + index


class FunctionsDataset(data.Dataset):

    def __init__(self,
                 root_dir: str,
                 split_file: str,
                 mode: str = "pairs",
                 format: str = "tokens",
                 split_subset: int = 0,
                 cache: bool = False):

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
        with open(path.join(self.root_dir, split_file)) as f:
            self.function_bases = list(map(lambda l: l.strip(), f.readlines()))

        if split_subset > 0:
            self.function_bases = self.function_bases[:split_subset]

        src_files: Dict[str, List[str]] = {}
        for fun_base in self.function_bases:
            paths = listdir(path.join(self.root_dir, fun_base))
            paths = set(filter(lambda p: p.endswith(required_extension), paths))
            paths = list(paths - tokenizing_errors[fun_base])
            paths = [path.join(self.root_dir, fun_base, fpath) for fpath in paths]
            src_files[fun_base] = paths

        if self.singles:
            self.src_files = list(chain(*src_files.values()))
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


    def __len__(self) -> int:
        return len(self.src_files)


    def __getitem__(self, index: int) -> DatasetItemType:
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
        if len(base_src_files) > 1:
            sample1_path, sample2_path = random.sample(base_src_files, 2)
        else:
            sample1_path, sample2_path = base_src_files[0], base_src_files[0]

        sample1 = self._parse_src(sample1_path)
        sample2 = self._parse_src(sample2_path)

        return sample1, sample2, index


    def _get_single(self, index: int) -> SingleType:
        src_file: str = self.src_files[index]
        base = path.dirname(src_file.replace(self.root_dir, ""))

        sample = self._parse_src(src_file)
        return sample, self.function_bases.index(base)


    def _get_negative_path(self, anchor) -> str:
        all_except_anchor: List[int] = list(self.indexes - {anchor})
        negative_base = self.function_bases[random.choice(all_except_anchor)]
        return random.choice(self.src_files[negative_base])


    def _parse_src(self, src_path: str) -> RepresentationType:
        if path.exists(src_path + ".pt"):
            try:
                return torch.load(src_path + ".pt")
            except:
                pass # fallback to standard parsing
        if self.graph:
            parsed_data = self._parse_graph(src_path)
        else:
            parsed_data = self._tokenize(open(src_path, "r").read())
        # cache
        if self.cache:
            torch.save(parsed_data, src_path + ".pt")
        return parsed_data


    def _parse_graph(self, src_path: str) -> geoData:
        with open(src_path, 'r') as f:
            nodes = list(map(int, f.readline().strip().split(",")))
            nodes = list(map(lambda x: node_attrs.index(x), nodes))

            edges = [[], []]
            edge_attrs = []
            for line in f:
                src, dst, tp = map(int, line.strip().split(","))
                edges[0].append(src)
                edges[1].append(dst)
                if not self.directed:
                    edges[0].append(dst)
                    edges[1].append(src)
                edge_attrs.append(tp)
                edge_attrs.append(tp)

        nodes = torch.LongTensor(nodes).reshape(-1, 1)
        edges = torch.LongTensor(edges)
        edge_attrs = torch.FloatTensor(edge_attrs).reshape(-1, 1)
        return geoData(x=nodes, edge_index=edges, edge_attr=edge_attrs)


    def _tokenize(self, code: str) -> torch.LongTensor:
        if self.tokens:
            tokens: List[str] = tokenize(code)
            return torch.LongTensor(tokens2idxs(tokens))
        parsed_code: str = parse(code)
        return torch.LongTensor(chars2idxs(parsed_code))