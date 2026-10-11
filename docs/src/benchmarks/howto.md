# How to Benchmark

The harness lives in the `benchmark/` submodule
([JuliaLegate/benchmarking](https://github.com/JuliaLegate/benchmarking)). It
compares cuNumeric.jl with CUDA.jl, JACC.jl and Dagger.jl, each in
its own process and environment. This page covers the essentials.

!!! tip
    For every option, config key and benchmark, see the harness
    [README](https://github.com/JuliaLegate/benchmarking/blob/main/README.md).

## Why not BenchmarkTools.jl?

cuNumeric operations are asynchronous: a call returns before the GPU finishes.
BenchmarkTools.jl, Chairmarks.jl or `@time` therefore under-report unless you
synchronize. The harness synchronizes after warmup, starts the clock, runs
`n_iter` iterations, synchronizes again, and stops the clock. BenchmarkTools'
allocation counts also miss Legate and CUDA buffers.

## Setup

```bash
git submodule update --init --recursive
cd benchmark
./deps-install/instantiate_projects.sh    # Julia environments
./deps-install/install_cupynumeric.sh     # cuPyNumeric Conda env
```

Setup develops cuNumeric.jl from the parent checkout. It uses `julia` from your
`PATH`, or `CUNUMERIC_BENCH_JULIA` if set.

### Container

The benchmark image comes with every environment set up, including cuPyNumeric:

```bash
docker pull ghcr.io/julialegate/benchmarking:latest
docker run -it --rm --gpus=all ghcr.io/julialegate/benchmarking:latest
```

This opens a shell in the benchmark checkout, ready for the commands below. Published tags are listed under
[packages](https://github.com/JuliaLegate/benchmarking/pkgs/container/benchmarking).

## Run

```bash
julia --project=. run.jl --config=configs/single_gpu/smoke.toml   # quick check
julia --project=. run.jl --config=configs/multi_gpu/gemm.toml --models=cunumeric,cudajl
julia --project=. run.jl --only=grayscott --models=cunumeric --fusion=both --dry-run
```

| Option | Meaning |
|---|---|
| `--config` | TOML config (default `configs/multi_gpu/all.toml`) |
| `--only` | benchmarks to run |
| `--models` | any of `cunumeric`, `cupynumeric`, `cudajl`, `jacc`, `dagger` |
| `--fusion` | `on`, `off` or `both` (cuNumeric only) |
| `--output` | result directory (default: timestamped under `results/`) |
| `--dry-run` | print the plan without running |

Each run launches a fresh worker with `LEGATE_CONFIG` set before Julia starts,
so GPU and CPU counts never change mid-process.

## Configs

```toml
[Global]
models = ["cunumeric", "cudajl"]
n_warmup = 2
n_iter = 10
n_trial = 5

[[gemm]]
T = "Float32"
gpus = [1, 2, 4, 8]
N = [43408, 54688, 68904, 86816]
```

- `gpus`, `cpus`, `N` and `M` pair up by position; `T` and `fusion` multiply.
- `auto_size = true` with `N` omitted fits the problem to `mem_frac` of the
  smallest GPU's memory.
- `single_gpu = true` sweeps problem size on one GPU instead of GPU count.

See `configs/single_gpu/` and `configs/multi_gpu/` for working examples.

## Results

Each run writes CSVs and a manifest to `results/<run-id>/` with mean time,
GFLOPS and a correctness check (`pass`, `fail` or `skipped`) per trial.

```bash
julia --project=. plotter/plot_results.jl results/<run-id> --format=pdf
```

For manual REPL timing, set `LEGATE_CONFIG` before starting Julia; see
[Hardware Configuration](../configuration/hardware.md).
