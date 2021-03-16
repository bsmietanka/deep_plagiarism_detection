from typing import List
from gst import match


# DIRTY HACK, RATATATAT
def tokens2letters(dictionary: List[str]):
    assert len(dictionary) < 100
    return {t:chr(60 + i) for i, t in enumerate(dictionary)}

class GSTSimilarity:

    def __init__(self, min_len: int = 3):
        self.min_len = min_len

    def __call__(self, f1: str, f2: str) -> float:
        matches = match(f1, "", f2, "", self.min_len)
        similarity = sum(m[2] for m in matches)
        return similarity / max(len(f1), len(f2))
