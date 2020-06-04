import os
import sys
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool, Queue
from functools import partial

from utils.r_tokenizer import parse, tokenize
from datasets import Dataset

# pattern = os.path.join(sys.argv[1], "**/*.R")
# fnames = glob(pattern, recursive=True)

# vocab = set()
# errors = []

# def worker(fname):
#     code = open(fname, "r").read()
#     try:
#         parsed = parse(code)
#         tokenized = tokenize(code)
#     except:
#         return 0, 0
#     return len(parsed), len(tokenized)


dataset_chars = Dataset(root_dir="/home/smietankab/private/thesis/data/alpha=1.0/n=500/p=0.1/r=0.1",
    class_ratio=0.5, tokens=False)
dataset_tokens = Dataset(root_dir="/home/smietankab/private/thesis/data/alpha=1.0/n=500/p=0.1/r=0.1",
    class_ratio=0.5, tokens=True)

def worker2(dataset_chars, dataset_tokens, idx):
    c1, c2 = dataset_chars[idx][:2]
    t1, t2 = dataset_tokens[idx][:2]
    return max(len(c1), len(c2)), max(len(t1), len(t2))

# max_chars = 0
# max_tokens = 0
# with Pool(8) as p:
#     for num_chars, num_tokens in tqdm(p.imap_unordered(worker, fnames), total=len(fnames)):
#         max_chars = max(max_chars, num_chars)
#         max_tokens = max(max_tokens, num_tokens)

worker_dataset = partial(worker2, dataset_chars, dataset_tokens)

max_chars = 0
max_tokens = 0
tot = len(dataset_chars)
with Pool(8) as p:
    for num_chars, num_tokens in tqdm(p.imap_unordered(worker_dataset, list(range(tot)), chunksize=100), total=tot):
        max_chars = max(max_chars, num_chars)
        max_tokens = max(max_tokens, num_tokens)

print(f"Max number of characters: {max_chars}")
print(f"Max number of tokens: {max_tokens}")
