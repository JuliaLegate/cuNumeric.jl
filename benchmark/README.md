# cuNumeric.jl benchmarks

This directory runs the same GPU benchmarks across cuNumeric.jl, cuPyNumeric,
CUDA.jl, JACC.jl, and Dagger.jl. Each model runs in an isolated process and
environment. Runs include correctness checks, trial progress, mean time, and
mean throughput with trial standard deviations.

## Setup

Instantiate the Julia environments once:

```bash
./instantiate_projects.sh
```

This develops the local cuNumeric package from `../` and CNPreferences from
`../lib/CNPreferences`. Set `CUNUMERIC_BENCH_JULIA` to select a different Julia
executable.

cuPyNumeric also needs its conda environment:

```bash
./install_cupynumeric.sh
```

Set `CUNUMERIC_BENCH_CONDA` if `conda` is not on `PATH`, or
`CUPYNUMERIC_ENV` to use an existing environment.

## Run

Use the smoke test for a quick end-to-end check:

```bash
julia --project=. run.jl --config=benchmarks_smoke.toml
```

Run the configured benchmark suite with:

```bash
julia --project=. run.jl
```

Useful filters:

```bash
julia --project=. run.jl --only=montecarlo
julia --project=. run.jl --only=gemm --models=cunumeric,cudajl,jacc,dagger
julia --project=. run.jl --only=grayscott --fusion=both
julia --project=. run.jl --only=montecarlo --dry-run
```

`--only` and `--models` accept comma-separated values. `--fusion` accepts
`on`, `off`, or `both`. Use `--verbose` for backend details.

## Configure

Benchmarks are declared in `benchmarks.toml`. Global values are inherited by
each benchmark block:

```toml
[Global]
models = ["cunumeric", "cupynumeric", "cudajl", "jacc", "dagger"]
n_warmup = 2
n_iter = 10
n_trial = 5
check_correctness = true
auto_size = true
mem_frac = 0.75

[[montecarlo]]
T = "Float32"
gpus = 1
cpus = 1
```

Set `N` and `M` explicitly for fixed problem sizes. When `auto_size = true`, an
omitted dimension is selected from `mem_frac` of the smallest visible GPU.
`T` and `fusion` form independent sweeps; `gpus`, `cpus`, `N`, and `M` are
zipped by position. A benchmark block may override `models`, `n_warmup`,
`n_iter`, or `n_trial`.

Native-library benchmarks such as GEMM require verified per-model scratch-space
bounds under `[workspace.<benchmark>]`; the planner reports any missing bound.

cuNumeric supports every registered benchmark. cuPyNumeric and CUDA.jl support
the non-accelerated array benchmarks; CUDA.jl is single-GPU. JACC and Dagger
currently support `montecarlo` and `gemm`.

## Results

Each run writes CSV files and a manifest to `results/<run-id>/`, then writes
plots to `plots/<run-id>/`. The manifest records resolved dimensions, memory
estimates, package versions, and worker status.

Timed iterations include synchronization but exclude initialization and warmup.
Each trial reports its mean milliseconds per iteration and GFLOP/s; the final
summary reports the mean and standard deviation across trials.

To plot existing CSV files:

```bash
julia --project=. plot_results.jl results/<run-id>
```
