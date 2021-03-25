from typing import Union, List
from scored_gst import token_comparison

InputType = Union[str, List[str]]
class GSTSimilarity:

    def __init__(self, min_len: int = 3):
        self.min_len = min_len


    def __call__(self, f1: InputType, f2: InputType) -> float:
        matches = token_comparison(f1, f2, self.min_len)
        similarity = sum(m["length"] for m in matches)
        return similarity / max(len(f1), len(f2))
