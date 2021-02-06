from typing import List

import rpy2.robjects as ro

from datasets.utils.r_tokens import r_tokens
from datasets.utils.vocab import characters


token2idx = {token:i for i, token in enumerate(r_tokens)}
idx2token = {i:token for i, token in enumerate(r_tokens)}
num_tokens = len(r_tokens)

char2idx = {char:i for i, char in enumerate(characters)}
idx2char = {i:char for i, char in enumerate(characters)}
num_chars = len(characters)
# pad idx could be handled better
pad_idx = 0


def escape_chars(string: str) -> str:
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

    code = escape_chars(code)
    # command = f"getParseData(parse(text=as.character(as.expression('{code}'))))$token"
    command = f"getParseData(parse(text='{code}', encoding='UTF-8'))$token"
    r_output = ro.r(command)
    tokens = list(r_output)
    tokens = [t.replace("'", "") for t in tokens]
    tokens = ["<SOS>", *tokens, "<EOS>"]
    return tokens


def parse(code: str) -> str:
    code = escape_chars(code)
    command = f"as.character(parse(text='{code}', keep.source=FALSE, encoding='UTF-8'))"
    r_output = ro.r(command)[0]
    # TODO: add <SOS> and <EOS> tokens
    return r_output


def chars2idxs(code: str) -> List[int]:
    return [char2idx.get(c, char2idx["UNKNOWN"]) for c in code]


def idxs2chars(idxs: List[int]) -> str:
    return "".join([idx2char[idx] for idx in idxs])


def tokens2idxs(tokens: List[str]) -> List[int]:
    return [token2idx[t] for t in tokens]


def idxs2tokens(idxs: List[int]) -> List[str]:
    return [idx2token[idx] for idx in idxs]


def normalize_idxs(idxs: List[List[int]]) -> List[List[float]]:
    num_tokens = len(token2idx) - 1
    return [[idx[0] / num_tokens] for idx in idxs]
