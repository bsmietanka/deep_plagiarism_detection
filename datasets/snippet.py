from glob import glob
from os import path
import editdistance
import pandas as pd
from tqdm import tqdm
from itertools import chain
from multiprocessing import Pool

data_root = "data"
csvs = glob(path.join(data_root, "**/!benchmark.csv"), recursive=True)

def worker_fun(csv_path):
    df = pd.read_csv(csv_path, sep=";", index_col=0)
    pairs = []
    for _, row in df.iterrows():
        f1 = row['functionName1']
        f2 = row['functionName2']
        if editdistance.eval(f1, f2) >= 3:
            pairs.append((f1, f2))
    return pairs

with Pool(12) as p:
    res_pairs = list(tqdm(p.imap_unordered(worker_fun, csvs), total=len(csvs)))

res_pairs = list(chain(*res_pairs))

print(len(res_pairs))
for i, p in enumerate(res_pairs[:50]):
    print(i, p)

### GRAPH ATTRS

def worker(src_file: str):
    edge_attrs = set()
    with open(src_file, "r") as f:
        node_attrs = set(map(int, f.readline().strip().split(",")))
        for line in f:
            _, _, tp = map(int, line.strip().split(","))
            edge_attrs.add(tp)
    return node_attrs, edge_attrs

data_root = "data/graph_functions/"
src_files = glob(path.join(data_root, "**/*.R.txt"), recursive=True)
with Pool(6) as p:
    res = list(tqdm(p.imap(worker, src_files), total=len(src_files)))

node_attrs = set()
edge_attrs = set()
for n, e in res:
    node_attrs.update(n)
    edge_attrs.update(e)

print(sorted(list(node_attrs)))
print(list(edge_attrs))
