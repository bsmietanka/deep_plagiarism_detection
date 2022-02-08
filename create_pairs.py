from pathlib import Path
from os import path
from random import sample
from typing import Set, Tuple

import pandas as pd
from tqdm import tqdm

graphs = True
balanced = False
n = 5000
test_dirs = list(Path(f"data/{'graphs' if graphs else 'text'}/").glob("**/!benchmark.csv"))
test_dirs = [d.parent for d in test_dirs]
graph_extension = "" if not graphs else ".txt"

with open("datasets/errors.txt") as f:
    files_w_errors = list(map(lambda l: l.strip(), f))

def is_negative(pair: Tuple[str, str], positives: Set[Tuple[str, str]]) -> bool:
    if pair not in positives:
        return True
    return False

def create_unbalanced_pairs(dir_path: Path, n: int = 1000):
    csv_path = dir_path / "!benchmark.csv"
    all_functions = list(dir_path.glob(f"*.R{graph_extension}"))
    all_functions = set(str(f) for f in all_functions) - set(files_w_errors)

    df = pd.read_csv(str(csv_path), sep=";", index_col=0)
    df["functionName1"] = str(dir_path) + "/" + df["functionName1"] + graph_extension
    df["functionName2"] = str(dir_path) + "/" + df["functionName2"] + graph_extension

    df = df[df["functionName1"].isin(all_functions)]
    df = df[df["functionName2"].isin(all_functions)]

    all_functions = list(Path(f) for f in sorted(all_functions))

    positive_names = df[["functionName1", "functionName2"]].to_dict("split")["data"]
    positive_names: Set[Tuple[str, str]] = set(tuple(map(lambda elem: elem.split("/")[-1], pair)) for pair in positive_names)
    negative, positive = [], []
    for i in range(n):
        for j in range(100):
            f1, f2 = sample(all_functions, 2)
            if (f1.name, f2.name) in negative or (f1.name, f2.name) in positive:
                continue
            if is_negative((f1.name, f2.name), positive_names):
                negative.append([f1.name, f2.name])
                break
            else:
                positive.append([f1.name, f2.name])
                break

    return positive, negative

def create_balanced_pairs(dir_path: Path, n: int = 1000):
    csv_path = dir_path / "!benchmark.csv"
    all_functions = list(dir_path.glob(f"*.R{graph_extension}"))
    all_functions = set(str(f) for f in all_functions) - set(files_w_errors)

    df = pd.read_csv(str(csv_path), sep=";", index_col=0)
    df["functionName1"] = str(dir_path) + "/" + df["functionName1"] + graph_extension
    df["functionName2"] = str(dir_path) + "/" + df["functionName2"] + graph_extension

    # breakpoint()
    df = df[df["functionName1"].isin(all_functions)]
    df = df[df["functionName2"].isin(all_functions)]

    all_functions = list(Path(f) for f in all_functions)

    if n > len(df):
        n = len(df)
    positive = df.sample(n)[["functionName1", "functionName2"]].to_dict("split")["data"]
    positive = [list(map(path.basename, p)) for p in positive]

    positive_names = df[["functionName1", "functionName2"]].to_dict("split")["data"]
    # breakpoint()
    positive_names: Set[Tuple[str, str]] = set(tuple(map(lambda elem: elem.split("/")[-1], pair)) for pair in positive_names)
    negative = []
    for i in range(n):
        for j in range(100):
            f1, f2 = sample(all_functions, 2)
            if is_negative((f1.name, f2.name), positive_names):
                negative.append([f1.name, f2.name])
                break

    return positive, negative

if balanced:
    create_pairs = create_balanced_pairs
else:
    create_pairs = create_unbalanced_pairs

for test_dir in tqdm(test_dirs):
    if (test_dir / "!pairs_unbalanced.csv").exists():
        continue
    pos, neg = create_pairs(test_dir, n=n)
    with (test_dir / "!pairs_unbalanced.csv").open("w") as f:
        for pair in pos:
            f.write(f"{pair[0]},{pair[1]},{1}\n")
        for pair in neg:
            f.write(f"{pair[0]},{pair[1]},{0}\n")
