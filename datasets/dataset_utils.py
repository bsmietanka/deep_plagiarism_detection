from collections import defaultdict
from multiprocessing import Pool
from os import path, walk
from typing import List, Tuple
from functools import partial
from pickle import dump

import pandas as pd
from tqdm import tqdm
from glob import glob


# TODO: document this better, hard to understand what is going on here

def parse_code_dir(code_dir: str, data_root: str) -> dict:
    """
    Args:
        code_dir (str): path relative to data_root
        data_root (str): assumes data_root path is ending with "/"
    Returns:
        dict: dictionary in a form of K: str, V: Set[str], each key connects
        to a set of plagiarized functions
    """
    data_root = path.join(data_root, "")
    files = defaultdict(set)
    csv_path = path.join(data_root, code_dir, "!benchmark.csv")
    df: pd.DataFrame = pd.read_csv(csv_path, sep=";", index_col=0)
    pairs = df[["functionName1", "functionName2"]].values
    # This loop will connect entries in a dict that are plagiarism of each other
    for p in pairs:
        p = [path.join(code_dir, name) for name in p]
        # NOTE: this loop is not always correct but because of how !benchmark.csv is specified
        # (ordered alphebatically first by functionName1, then by functionName2, 
        # every plagiarism pair connected to each other) it should always work
        files[p[1]] = files[p[0]]
    fnames = glob(path.join(data_root, code_dir, "*.R"))
    # This loop will fill the structure with values
    for f in fnames:
        f = f.replace(data_root, "")
        # if f is not in files than f has no plagiarized function in this directory
        files[f].add(f)
    return files

def get_code_dirs(data_root: str) -> List[str]:
    """
    Get list of directories that contain .R files.
    Directories paths are relative to data_root.
    eg. data_root = "data/", data contains directories: 1, 2, 3
    """
    if not path.isdir(data_root):
        raise ValueError(f"data_root parameter is not a directory")
    data_root = path.join(data_root, "")
    dirs = set()
    for root, _, files in walk(data_root):
        if any(f.endswith(".R") for f in files):
            dirs.add(root.replace(data_root, ""))
    return list(dirs)


def merge_dir_results(dicts: Tuple[dict, dict]) -> dict:
    """Merge two dictionaries in a form Dict[str, Set[str]], where
    keys are R source code function paths, and values are connected
    plagiarized functions

    Args:
        dicts (Tuple[dict, dict])
    Returns:
        dict: merged dict
    """
    first = dicts[0]
    second = dicts[1]
    if len(first) == 0:
        return second
    # basenames = set(path.basename(p) for p in second)
    files2 = {path.basename(k): k for k in second}
    basenames = set(files2.keys())

    # common = files1 & files2
    # # NOTE: what is wrong with this function?
    # if "dplyr_id.R" in common:
    #     for_logging = list(filter(lambda f: f.endswith("dplyr_id.R"), second.keys()))
    #     logging.error(for_logging)

    common = []
    for f1 in first:
        base_name = path.basename(f1)
        if base_name in basenames:
            common.append(f1)
            basenames.remove(base_name)

    res = first
    res.update(second)
    for f in common:
        duplicate_path = files2[path.basename(f)]
        fnames1 = res[f]
        fnames2 = res[duplicate_path]
        new = fnames1 | fnames2
        for fname in new:
            res[fname] = new

    return res


def parse_dataset(data_root: str) -> dict:
    code_dirs = get_code_dirs(data_root)
    res = defaultdict(set)
    worker = partial(parse_code_dir, data_root=data_root)
    with Pool(8) as p:
        res = list(tqdm(p.imap_unordered(worker, code_dirs, 2),
                      total=len(code_dirs),
                      desc="Parsing folders"))
    # for c in tqdm(code_dirs):
    #         dir_res = parse_code_dir(c, data_root)
        while len(res) != 1:
            worker_iterable = [(res[2 * i], res[2 * i + 1]) for i in range(len(res) // 2)]
            rest = res[(len(res) // 2) * 2:]
            res = list(tqdm(p.imap_unordered(merge_dir_results, worker_iterable), total=len(worker_iterable)))
            res.extend(rest)

    return res[0]


if __name__ == "__main__":
    ds = ["/home/smetana/private/thesis/data/alpha=2.0/"]
    for d in ds:
        d = path.join(d, "")
        res = parse_dataset(d)

        reference = glob(path.join(d, "**/*.R"), recursive=True)
        reference = [p.replace(d, "") for p in reference]
        sanity_check = set(res.keys()) - set(reference)
        if len(sanity_check) != 0:
            print(sanity_check)
            exit(1)
        # print(set(res.keys()) - set(reference))

        # print number of all unique sets
        set_of_sets = set(frozenset(v) for v in res.values())
        print("Number of plagiarism:", len(set_of_sets))
        print("Total number of functions:", len(reference))
        if sum(len(s) for s in set_of_sets) != len(reference):
            exit(1)
        list_of_lists = [list(v) for v in set_of_sets]
        import editdistance
        for l in tqdm(list_of_lists):
            x = list(set(path.basename(p) for p in l))
            dist_matrix = [[editdistance.eval(ref, p) for p in x] for ref in x]
            num_different = [sum(dist >= 3 for dist in dist_vec) for dist_vec in dist_matrix]
            best = min(num_different)
            if best >= 0.9 * len(x):
                print("##############\n", best, len(x), x[:500])
                # for i, d in enumerate(dists):
                #     if d >= 3:
                #         print(x[i])

        print("Saving results")
        dump(res, open(path.join(d, "triplets.pickle"), "wb"))
        dump(list_of_lists, open(path.join(d, 'plagiarism_lists.pickle'), 'wb'))
