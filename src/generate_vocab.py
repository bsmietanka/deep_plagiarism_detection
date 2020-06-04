import os
import sys
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool, Queue

from utils.r_tokenizer import parse

pattern = os.path.join(sys.argv[1], "**/*.R")
fnames = glob(pattern, recursive=True)

vocab = set()
errors = []

def worker(fname):
    code = open(fname, "r").read()
    try:
        parsed = parse(code)
    except:
        return None, fname
    return set(parsed), fname

unusual_chars_files = []
unusual_chars = set("£±½×ΓΔΘΛΞΠΣΥΦΨΩαβγδεζηθικλμνξπρςστυφχψωϑϖϱϵ٪‘’⁄≤≥年日月﹪")
with Pool(8) as p:
    for res in tqdm(p.imap_unordered(worker, fnames), total=len(fnames)):
        if type(res[0]) is set:
            if not unusual_chars.isdisjoint(res[0]):
                unusual_chars_files.append(res[1])
            vocab.update(res[0])
        else:
            errors.append(res[1])

characters = "\n".join(sorted(list(vocab)))
with open("vocab.txt", 'w') as f:
    f.writelines(characters)

with open("errors.txt", 'w') as f:
    f.writelines("\n".join(errors))

with open("unusual.txt", 'w') as f:
    f.writelines("\n".join(unusual_chars_files))

