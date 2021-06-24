import json
from os import walk, path

from tqdm import tqdm

from learn_metric import main

cuda0 = []
cuda1 = []
for root, dirs, files in walk("configs"):
    for fname in files:
        fname = path.join(root, fname)
        with open(fname, "r") as f:
            params = json.load(f)
        if params["device"] == "cuda:1":
            cuda1.append(fname)
        elif params["device"] == "cuda:0":
            cuda0.append(fname)
        else:
            raise ValueError

def worker(configs):
    for cpath in tqdm(configs):
        with open("cuda1_task_results.txt", "a") as f:
            try:
                best_acc = main(cpath)
                s = "OK"
            except Exception as e:
                best_acc = "-"
                s = "FAIL"
            f.write(f"{cpath},{s},{best_acc}\n")


status = worker(cuda1)
