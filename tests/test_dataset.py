from os import listdir, path
from glob import glob

from torch_geometric.data import Data as geoData
import torch

from datasets.functions_dataset import FunctionsDataset


data_root = "data/functions"
graph_data_root = "data/graph_functions"

def test_tokenized_dataset():
    dataset = FunctionsDataset(data_root, listdir(data_root))
    for i in range(len(dataset)):
        items = dataset[i]
        assert len(items) == 4
        anchor, positive, negative, base_name = items
        assert isinstance(anchor, torch.Tensor)
        assert isinstance(positive, torch.Tensor)
        assert isinstance(negative, torch.Tensor)
        assert base_name == dataset.function_bases[i]

def test_char_dataset():
    dataset = FunctionsDataset(data_root, listdir(data_root), triplets=False, tokens=False)
    for i in range(len(dataset)):
        idx = i % (len(dataset) // 2)
        items = dataset[i]
        assert len(items) == 4
        sample1, sample2, plagiarism, base_name = items
        assert isinstance(sample1, torch.Tensor)
        assert isinstance(sample2, torch.Tensor)
        assert plagiarism == 1. or plagiarism == 0.
        assert base_name == dataset.function_bases[idx]

def test_graph_dataset():
    dataset = FunctionsDataset(graph_data_root, listdir(graph_data_root), graphs=True)
    for i in range(len(dataset)):
        items = dataset[i]
        assert len(items) == 4
        anchor, positive, negative, base_name = items
        assert isinstance(anchor, geoData)
        assert isinstance(positive, geoData)
        assert isinstance(negative, geoData)
        assert base_name == dataset.function_bases[i]

def test_tokenizing():
    dataset = FunctionsDataset(data_root, listdir(data_root))
    src_files = list(filter(path.isfile, glob(path.join(data_root, "**/*.R"), recursive=True)))
    for src_file in src_files:
        d = dataset._parse_graph(src_file)
        assert isinstance(d, torch.Tensor)

def test_parsing():
    dataset = FunctionsDataset(data_root, listdir(data_root), triplets=False, tokens=False)
    src_files = list(filter(path.isfile, glob(path.join(data_root, "**/*.R"), recursive=True)))
    for src_file in src_files:
        d = dataset._parse_graph(src_file)
        assert isinstance(d, torch.Tensor)

def test_parsing_graph():
    dataset = FunctionsDataset(graph_data_root, listdir(graph_data_root), graphs=True)
    src_files = list(filter(path.isfile, glob(path.join(data_root, "**/*.R"), recursive=True)))
    for src_file in src_files:
        d = dataset._parse_graph(src_file)
        assert isinstance(d, geoData)

