import os
from typing import List
import warnings

import rpy2.robjects as ro
from rpy2.rinterface import RRuntimeWarning

from utils.r_tokens import r_tokens


token2idx = {token:i for i, token in enumerate(r_tokens)}
idx2token = {i:token for i, token in enumerate(r_tokens)}
vocab_size = len(token2idx)

def escape(string: str) -> str:
    return string.translate(str.maketrans({
        # "-":  r"\-",
        # "]":  r"\]",
        "\\": r"\\",
        "'": r"\'",
        # "^":  r"\^",
        # "$":  r"\$",
        # "*":  r"\*",
        # ".":  r"\."
        }))

def tokenize(code: str) -> List[str]:
    # tokens = list(ro.r(f'suppressWarnings(getParseData(parse(text=as.character(as.expression("{code}"))))$token)'))

    code = escape(code)
    command = f"getParseData(parse(text=as.character(as.expression('{code}'))))$token"
    r_output = ro.r(command)
    tokens = list(r_output)
    tokens = [t.replace("'", "") for t in tokens]
    return tokens

def tokens2idxs(tokens: List[str]) -> List[int]:
    return [[token2idx[t]] for t in tokens]

def idxs2tokens(idxs: List[int]) -> List[str]:
    return [[idx2token[idx]] for idx in idxs]

def normalize_idxs(idxs: List[List[int]]) -> List[List[float]]:
    num_tokens = len(token2idx) - 1
    return [[idx[0] / num_tokens] for idx in idxs]

if __name__ == "__main__":
    print(tokenize('pi'))