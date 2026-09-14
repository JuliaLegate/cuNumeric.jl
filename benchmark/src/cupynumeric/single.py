# cuPyNumeric worker; the model adapter sets LEGATE_CONFIG before launch.
# Args: <gpus> <name> <T> <N> <M> <n_iter> <n_warmup> <n_trial>
#       [check_correctness] [n_correctness_iter] [flops]
# flops comes from the Julia orchestrator (same total_flops as the kernel file).
import os
import sys
import time as walltime

if os.environ.get("CUNUMERIC_BENCH_ACTIVE_MODEL") != "cupynumeric":
    raise RuntimeError(
        "single.py must be launched by run_benchmark.sh with --model=cupynumeric"
    )

# Make `core` and the `benchmarks` package importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import MOD, parse_type, trial, save_result, _mean, _std
from benchmarks import BENCHMARKS  # import populates BENCHMARKS


def show_progress(name, completed, total, last_time_ms, last_gflops, started):
    width = 40
    filled = width * completed // total
    bar = "█" * filled + " " * (width - filled)
    percent = 100 * completed // total
    elapsed = int(walltime.monotonic() - started)
    sys.stderr.write(
        f"\r{name} trials: {percent:3d}%|{bar}| Time: 0:{elapsed // 60:02d}:{elapsed % 60:02d}"
    )
    sys.stderr.flush()
    if completed == total:
        sys.stderr.write(
            f"\n                 Completed trials: {completed}/{total}"
            f"\n   Last trial mean (ms/iteration): {last_time_ms:.5f}"
            f"\n               Last trial GFLOP/s: {last_gflops:.5f}\n"
        )
        sys.stderr.flush()


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
    verbose = os.environ.get("CUNUMERIC_BENCH_VERBOSE", "0") == "1"
    supports_correctness = hasattr(bench, "check_correctness")
    if verbose and check_correctness and supports_correctness:
        print(
            "Correctness check: reference=CPU, "
            f"dimensions={bench.correctness_dims()[0]}×{bench.correctness_dims()[1]}"
        )
    correctness = (
        bench.check_correctness()
        if check_correctness and supports_correctness
        else "skipped"
    )

    if verbose:
        print(
            f"[{MOD}] {name} benchmark ({T_str}) on {N}x{M} for {n_iter} "
            f"iterations ({n_warmup} warmup) x {n_trial} trials; "
            + ("per-iteration synchronization"
               if getattr(bench, "fence_each_iteration", True)
               else "batch synchronization")
        )

    times_ms, gflops = [], []
    progress_started = walltime.monotonic()
    for trial_index in range(1, n_trial + 1):
        t, g = trial(bench, n_warmup, n_iter, flops)
        times_ms.append(t)
        gflops.append(g)
        show_progress(name, trial_index, n_trial, t, g, progress_started)

    print(f"[{MOD}] Correctness: {correctness}")
    print(
        f"[{MOD}] Mean time: {_mean(times_ms):.5f} ± {_std(times_ms):.5f} ms "
        "(trial SD)"
    )
    print(
        f"[{MOD}] Mean throughput: {_mean(gflops):.5f} ± {_std(gflops):.5f} "
        "GFLOP/s (trial SD)"
    )

    save_result(bench.name, bench.dims(), gpus, times_ms, gflops, correctness)


if __name__ == "__main__":
    main()
