# How to Benchmark

The benchmark harness lives in `benchmark/` at the repo root. Configs are declared in `benchmarks.toml`. `run.jl` expands those configs and launches one worker process per run. Workers never share a GPU runtime within a measurement.

> [!WARNING]
> We do not commit to maintaining the benchmark scripts forever. The harness evolves with the package. The ideas here (declare configs in TOML, one process per run, time with Legate fences) should still apply even if file names move.

## Why not BenchmarkTools.jl?

cuNumeric ops are asynchronous. A Julia call usually returns before the GPU work finishes, so tools like [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl), [Chairmarks.jl](https://github.com/LilithHafner/Chairmarks.jl), or even a bare `Base.@time` will under-report time unless you force a fence. The harness uses `get_time_microseconds` / `get_time_nanoseconds`, which block until preceding Legate work completes. Allocations reported by BenchmarkTools will also not reflect Legion / CUDA buffers.

## Quick start

From the repo:

```bash
cd benchmark
julia --project=. run.jl
```

With no extra args, `run.jl` reads `benchmarks.toml` and runs every expanded config. It develops `CNPreferences` and `cuNumeric` from the parent checkout, then for each config:

1. Sets the broadcast-fusion preference if needed and precompiles
2. Calls `run_benchmark.sh`, which exports `LEGATE_CONFIG` from `--gpus` / `--cpus` **before** Julia starts
3. Launches `src/single.jl` for the cuNumeric backend (and optional comparison backends)

Add `-v` / `--verbose` for more plumbing output.

## `benchmarks.toml`

`[Global]` sets defaults shared by every run:

```toml
[Global]
n_warmup = 5
n_iter = 1000
n_trial = 5
cupynumeric = true   # also run Python cupynumeric (needs install_cupynumeric.sh)
cuda = false         # also run CUDA.jl (single-GPU configs only)
check_correctness = true
n_correctness_iter = 5
```

- `n_warmup`: untimed iterations (hide compile / first-touch cost)
- `n_iter`: timed iterations per trial (build task queue depth)
- `n_trial`: independent trials; mean ± stddev across trials is what gets printed / saved
- `cupynumeric` / `cuda`: optional comparison backends
- `check_correctness`: one CPU-reference check per config (not per timed iter), recorded in the CSV

Each `[[name]]` block is a registered benchmark (`gemm`, `montecarlo`, `dmd_baseline`, `dmd_lifetimes`, `grayscott_baseline`, `grayscott_lifetimes`, `poisson_fft`, …). Names must match what `src/benchmarks/*.jl` registers.

DMD's `N` is the number of spatial degrees of freedom (rows of the snapshot matrix), not a grid side length. The SVD is of the tall-skinny `N × (M-1)` matrix `X1`. Thin SVD plus the rank-`r` lift is `Θ(N)` when `M` and `r` are fixed, so weak scaling is `N ∝ P` (same idea as Monte Carlo, not GEMM's `N ∝ P^{1/3}`). The flop count is in `src/benchmarks/dmd.jl`.

`poisson_fft` solves ``M`` independent periodic Poisson problems on an ``N \times N`` grid (FFT, divide by ``-|k|^2``, inverse FFT). The transform is over the last two axes, so the leading batch axis can split across GPUs. Weak scaling is ``M \propto P`` with ``N`` fixed. A single all-axes 2-d `fft` of one grid is single-GPU and would not scale that way.

```toml
[[gemm]]
T = ["Float32"]
gpus = [1, 2, 4, 8]
cpus = 16
N = [20000, 25200, 31752, 40000]
M = [20000, 25200, 31752, 40000]
```

### How lists expand

Any of `T`, `fusion`, `gpus`, `cpus`, `N`, `M` may be a scalar or a list.

- **`T` and `fusion` multiply.** The sweep runs once per type and once per fusion setting (`fusion = [true, false]` sweeps both).
- **`gpus`, `cpus`, `N`, `M` zip** in lockstep. Element `i` of each is paired together. A scalar broadcasts to every position.

```toml
[[gemm]]
T    = ["Float64", "Float32"]   # multiplies
gpus = [1, 2, 4]
cpus = 2                        # zip -> (1,2,150,150), (2,2,300,300), (4,2,600,600)
N    = [150, 300, 600]
M    = [150, 300, 600]
```

That is 2 types × 3 sweep points = **6 runs**.

`fusion` toggles cuNumeric broadcast fusion (`true`/`false` or `"on"`/`"off"`,
default `true`). Comparison backends ignore fusion and run once (on the fused
pass), not per variant. Entries ending in `_accelerated` are cuNumeric-only.
The Gray-Scott function, `begin`, `let`, and expression entries compare the
four `@accelerate` scope contracts on the same step; `dmd_accelerated` applies
the recommended function form to the DMD projection.

Gotcha: when `T = ["Float32", "Float64"]` and a length-2 `N`/`M` sweep you get all **4** combinations, not a paired `Float32 -> N[1]`. To pin a type to a size, use separate `[[name]]` blocks.

## One-off runs

You can dispatch a single config without editing the TOML:

```bash
julia --project=. run.jl <gpus> <cpus> <name> <T> <N> <M> <n_iter> <n_warmup> <n_trial> [fusion]
```

Example:

```bash
julia --project=. run.jl 1 16 gemm Float32 20000 20000 1000 5 5 true
```

`run.jl` still goes through `run_benchmark.sh` so Legate sees the right GPU/CPU count at process start.

## Comparison backends

- **CUDA.jl:** set `cuda = true` in `[Global]`. Only runs when `gpus == 1`.
- **cupynumeric (Python):** set `cupynumeric = true`, then build a matching conda env once:

```bash
./install_cupynumeric.sh   # creates cupynumeric-bench-<major.minor>
```

`run.jl` picks the env from the resolved `cupynumeric_jll` version. Override with `CUPYNUMERIC_ENV`.

## Results and timing

Each worker prints mean ± stddev run time (ms) and GFLOPS, plus a correctness tag (`pass` / `fail` / `skipped`). CSVs append under `benchmark/results/`.

Unfused cuNumeric runs are labeled and saved separately (for example `cunumeric_nofusion`) so they stay a distinct series from fused runs.

## Hardware notes

`LEGATE_CONFIG` must be set before Julia / Legate starts. The harness does that for you via `run_benchmark.sh`. For manual REPL experiments, see [Hardware Configuration](../configuration/hardware.md). Do not expect to change GPU count mid-session without restarting Julia.
