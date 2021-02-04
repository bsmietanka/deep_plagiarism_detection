## performance of loading cached pickled tensors and parsing code with R each time

In [19]: %timeit torch.LongTensor(tokens2idxs(tokenize(open(ex, "r").read())))
Out: 6.62 ms ± 253 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

In [20]: %timeit torch.LongTensor(chars2idxs(parse(open(ex, "r").read())))
Out: 3.03 ms ± 316 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

In [21]: %timeit torch.load("tokens.pt")
Out: 87.7 µs ± 2.95 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

In [22]: %timeit torch.load("chars.pt")
Out: 88.6 µs ± 365 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
