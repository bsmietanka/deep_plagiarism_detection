from collections import defaultdict
import logging
from multiprocessing import Pool
from os import path, walk, listdir
from typing import List, Union

from multiprocessing_logging import install_mp_handler
import pandas as pd
from tqdm import tqdm
from glob import glob

install_mp_handler()


def parse_code_dir(code_dir: str):
    files = defaultdict(set)
    csv_path = path.join(code_dir, "!benchmark.csv")
    df: pd.DataFrame = pd.read_csv(csv_path, sep=";", index_col=0)
    pairs = df[["functionName1", "functionName2"]].values
    for p in pairs:
        p = [path.join(code_dir, name) for name in p]
        files[p[1]] = files[p[0]]
    fnames = glob(path.join(code_dir, "*.R"))
    for f in fnames:
        files[f].add(f)
    return files

def get_code_dirs(data_dirs: Union[str, List[str]]) -> List[str]:
    dirs = set()
    if type(data_dirs) is str:
        data_dirs = [data_dirs]
    for data_dir in data_dirs:
        for root, _, files in walk(data_dir):
            if any(f.endswith(".R") for f in files):
                dirs.add(root)
    return list(dirs)

def merge_dir_results(first, second):
    dir2 = path.dirname(next(iter(second)))
    files1 = set(path.basename(p) for p in first)
    files2 = set(path.basename(p) for p in second)

    common = files1 & files2
    if "dplyr_id.R" in common:
        for_logging = list(filter(lambda f: f.endswith("dplyr_id.R"), second.keys()))
        logging.error(for_logging)

    main_common = []
    for base_name in common:
        for f in first:
            if f.endswith(base_name):
                main_common.append(f)
                break

    res = first
    res.update(second)
    for f in main_common:
        duplicate_path = path.join(dir2, path.basename(f))
        if f not in res or duplicate_path not in res:
            logging.error(f"{f},{f not in res},{duplicate_path},{duplicate_path not in res},{duplicate_path not in second}")
            continue
        fnames1 = res[f]
        fnames2 = res[duplicate_path]
        new = fnames1 | fnames2
        for fname in new:
            if fname not in res:
                logging.error(f"Error: {fname}")
            res[fname] = new

    return res


def parse_dataset(data_dirs: Union[str, List[str]]):
    code_dirs = get_code_dirs(data_dirs)
    res = None
    with Pool(8) as p:
        for dir_res in tqdm(p.imap_unordered(parse_code_dir, code_dirs, 2),
                      total=len(code_dirs),
                      desc="Parsing folders"):
    # for c in tqdm(code_dirs):
    #         dir_res = parse_code_dir(c)
            if res is None:
                res = dir_res
            else:
                res = merge_dir_results(res, dir_res)

    return res

def glob_dirs(dirs):
    if type(dirs) is str:
        dirs = [dirs]
    res = set()
    for d in dirs:
        res.update(glob(path.join(d, "**/*.R"), recursive=True))
    return list(res)


if __name__ == "__main__":
    ds = ["/home/smietankab/private/thesis/data/"]
    for d in ds:
        print(d)
        res = parse_dataset(d)

        from pickle import dump


        reference = glob_dirs(d)
        print(len(res))
        print(len(reference))
        print(set(res.keys()) - set(reference))

        if len(res) - len(reference) == 0:
            dump(res, open(path.join(d, "triplets.pickle"), "wb"))
