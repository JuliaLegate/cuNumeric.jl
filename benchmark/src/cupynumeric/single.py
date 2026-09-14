# cuPyNumeric worker; the model adapter sets LEGATE_CONFIG before launch.
# Args: <gpus> <name> <T> <N> <M> <n_iter> <n_warmup> <n_trial>
#       [check_correctness] [n_correctness_iter] [flops]
# flops comes from the Julia orchestrator (same total_flops as the kernel file).
import os
import sys

if os.environ.get("CUNUMERIC_BENCH_ACTIVE_MODEL") != "cupynumeric":
    raise RuntimeError(
        "single.py must be launched by run_benchmark.sh with --model=cupynumeric"
    )

# Make `core` and the `benchmarks` package importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import MOD, parse_type, trial, save_result, _mean, _std
from benchmarks import BENCHMARKS  # import populates BENCHMARKS


def main():
    gpus = int(sys.argv[1])
    name = sys.argv[2]
    T_str = sys.argv[3]
    N = int(sys.argv[4])
    M = int(sys.argv[5])
    n_iter = int(sys.argv[6])
    n_warmup = int(sys.argv[7])
    n_trial = int(sys.argv[8])
    if len(sys.argv) < 12:
        raise SystemExit(
            "single.py args: <gpus> <name> <T> <N> <M> <n_iter> <n_warmup> "
            "<n_trial> <check> <n_correctness_iter> <flops>"
        )
    flops = float(sys.argv[11])
    check_correctness = sys.argv[9].lower() == "true"

    if name not in BENCHMARKS:
        raise ValueError(
            f"No benchmark registered for '{name}'. Known: {', '.join(sorted(BENCHMARKS))}"
        )
    T = parse_type(T_str)
    bench = BENCHMARKS[name](T, N, M)
    correctness = bench.check_correctness() if check_correctness else "skipped"

    print(
        f"[{MOD}] {name} benchmark ({T_str}) on {N}x{M} for {n_iter} "
        f"iterations ({n_warmup} warmup) x {n_trial} trials; "
        + ("per-iteration synchronization" if getattr(bench, "fence_each_iteration", True)
           else "batch synchronization")
    )

    times_ms, gflops = [], []
    for _ in range(n_trial):
        t, g = trial(bench, n_warmup, n_iter, flops)
        times_ms.append(t)
        gflops.append(g)

    print(f"[{MOD}] Mean Run Time: {_mean(times_ms):.5f} ± {_std(times_ms):.5f} ms")
    print(f"[{MOD}] FLOPS: {_mean(gflops):.5f} ± {_std(gflops):.5f} GFLOPS")
    print(f"[{MOD}] Correctness: {correctness}")

    save_result(bench.name, bench.dims(), gpus, times_ms, gflops, correctness)


if __name__ == "__main__":
    main()
