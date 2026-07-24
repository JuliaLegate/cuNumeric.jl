import os
import math

import cupynumeric as np
from legate.timing import time  # blocks on preceding legate ops; returns microseconds

MOD = "cupynumeric"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

DTYPES = {"Float32": np.float32, "Float64": np.float64}


def parse_type(s):
    if s not in DTYPES:
        raise ValueError(f"Unsupported type '{s}'. Known: {', '.join(DTYPES)}")
    return DTYPES[s]


BENCHMARKS = {}


def register_benchmark(key, cls):
    BENCHMARKS[key] = cls


def trial(bench, n_warmup, n_iter):
    state = bench.initialize()
    start = None
    for idx in range(n_warmup + n_iter):
        if idx == n_warmup:
            start = time()
        bench.run(state)
    total_us = time() - start

    mean_time_ms = total_us / (n_iter * 1e3)
    gflops = bench.total_flops() / (mean_time_ms * 1e6)
    return mean_time_ms, gflops


def _mean(x):
    return sum(x) / len(x)


def _std(x):
    if len(x) < 2:
        return 0.0
    m = _mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1))


def save_result(name, dims, gpus, times_ms, gflops):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    N, M = dims
    path = os.path.join(RESULTS_DIR, f"{name}_{MOD}.csv")
    with open(path, "a") as io:
        for i, (t, g) in enumerate(zip(times_ms, gflops), start=1):
            io.write(f"{MOD},{gpus},{N},{M},{i},{t:.6f},{g:.6f}\n")
