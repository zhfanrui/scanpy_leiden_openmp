from __future__ import annotations

import argparse
import time

import numpy as np
import scipy.sparse as sp

from scanpy_leiden_openmp import leiden


def make_random_graph(n: int, degree: int, seed: int) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(n), degree)
    cols = rng.integers(0, n, size=n * degree)
    data = np.ones(n * degree, dtype=np.float64)
    graph = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    graph = graph.maximum(graph.T)
    return graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=10_000)
    parser.add_argument("--degree", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=-1)
    args = parser.parse_args()

    graph = make_random_graph(args.nodes, args.degree, args.seed)

    t0 = time.perf_counter()
    labels = leiden(graph, backend="openmp", strict=False, n_threads=args.threads)
    t1 = time.perf_counter()

    unique = len(np.unique(labels))
    print(f"nodes={args.nodes} degree={args.degree} communities={unique} elapsed={t1 - t0:.4f}s")


if __name__ == "__main__":
    main()
