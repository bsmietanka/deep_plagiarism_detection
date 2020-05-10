import pandas as pd
import os
import sys
import traceback

from tqdm import tqdm

from src.utils.r_tokenizer import tokenize

lengths = []
count = 0
errors = list()
for root, dirs, files in tqdm(os.walk('data')):
    if count > 100:
        break
    for f in files:
        if f[-2:] != '.R':
            continue
        count += 1
        code = open(os.path.join(root, f), 'r').read()
        try:
            tokens = tokenize(code)
            # lengths.append(len(tokens))
        except:
            errors.append(root + "/" + f + '\n')

with open('errors.txt', 'w') as f:
    f.writelines(errors)
# print(max(lengths))
# print(len(lengths))

with open('errors.txt', 'r') as f:
    files = [line.strip() for line in f.readlines()]

to_fix = []
for fname in files:
    code = open(fname, 'r').read()
    try:
        tokenize(code)
    except:
        if "unexpected '='" in traceback.format_exc():
            to_fix.append(fname + "\n")

with open('to_fix.txt', 'w') as f:
    for fname in to_fix:
        code = open(fname.strip(), 'r').read()
        try:
            tokenize(code)
        except:
            err_msg: str = traceback.format_exc()
            token = "<text>:"
            start = err_msg.find("<text>:")
            line, column = err_msg[start + len(token):start + len(token) + 10].split(":")[:2]
            f.write(f"{fname.strip()},{line},{column}\n")

with open('to_fix.txt', 'r') as f:
    lines = [line.strip() for line in f.readlines()]
    rows = [tuple(line.split(',')) for line in lines]
    rows = [tuple([row[0], int(row[1]), int(row[2])]) for row in rows]

for row in rows:
    file_content = open(row[0], 'r').readlines()
    line_to_change = file_content[row[1] - 1]
    beg, rest = line_to_change[:row[2] - 1], line_to_change[row[2] - 1:]
    rest = rest.replace("=", "<-", 1)
    file_content[row[1] - 1] = beg + rest
#     open(row[0], 'w').writelines(file_content)


with open('errors.txt', 'r') as f:
    files = [line.strip() for line in f.readlines()]

# files = ["data/alpha=1.5/n=500/p=0.25/r=0.3/2/grid_legendGrob2.R"]
errors = []
for fname in files:
    code = open(fname, 'r').read()
    # print(code)
    try:
        tokenize(code)
    except:
        errors.append(fname + "\n")

print(len(files))
print(len(errors))
with open('errors.txt', 'w') as f:
    f.writelines(errors)