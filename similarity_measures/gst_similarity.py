from multiprocessing import Pool
from typing import Union, List

from tqdm import tqdm
from gst import match
# from scored_gst import token_comparison


# DIRTY HACK, RATATATAT
def tokens2letters(dictionary: List[str]):
    assert len(dictionary) < 100
    return {t:chr(60 + i) for i, t in enumerate(dictionary)}

# def __call__(self, f1: str, f2: str) -> float:
#         matches = match(f1, "", f2, "", self.min_len)
#         similarity = sum(m[2] for m in matches)

def gst_similarity(f1, f2, l: int):
    dictionary = list(set(f1) | set(f2))
    matching = tokens2letters(dictionary)
    f1 = "".join(matching[elem] for elem in f1)
    f2 = "".join(matching[elem] for elem in f2)
    matches = match(f1, "", f2, "", l)
    return sum(m[2] for m in matches) / max(len(f1), len(f2))

# def gst_similarity(f1, f2, l: int):
#     matches = token_comparison(f1, f2, l)
#     similarity = sum(m["length"] for m in matches)
#     return similarity / max(len(f1), len(f2))

InputType = Union[str, List[str], List[List[str]]]
class GSTSimilarity:

    def __init__(self, min_len: int = 3):
        self.min_len = min_len
        self.p = Pool(20)

    def __call__(self, f1: InputType, f2: InputType) -> Union[float, List[float]]:
        if isinstance(f1, list) and isinstance(f1[0], list):
            res = [self.p.apply_async(gst_similarity, [x1, x2, self.min_len]) for x1, x2 in zip(f1, f2)]
            return [r.get() for r in res]
        return gst_similarity(f1, f2, self.min_len)
