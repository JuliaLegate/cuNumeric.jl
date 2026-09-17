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

Run the focused cuNumeric, cuPyNumeric, and Dagger Gray-Scott weak-scaling
check on 1, 2, 4, and 8 GPUs with:

```bash
julia --project=. run.jl --config=benchmarks_grayscott_multigpu.toml --verbose
```

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

For JACC and Dagger, `run_benchmark.sh` restricts each worker to the requested
GPU count. It selects the first `gpus` entries from an existing
`CUDA_VISIBLE_DEVICES` scheduler mask, or uses logical devices starting at zero
when no mask is provided.

Native-library benchmarks such as GEMM require verified per-model scratch-space
bounds under `[workspace.<benchmark>]`; the planner reports any missing bound.

cuNumeric supports every registered benchmark. CUDA.jl and cuPyNumeric run the
non-accelerated array benchmarks (CUDA.jl is single-GPU). JACC and Dagger have
native `montecarlo`, `gemm`, `grayscott`, and `cg`.

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

### Conjugate gradient

`cg` is the default variant and runs on every model; on cuNumeric it applies
`@accelerate` to each update. `cg_plain` is the cuNumeric variant without
`@accelerate`, for the accelerate comparison. The generic solver is shared by the
array workers (`src/benchmarks/cg.jl`); JACC and Dagger have native versions.

```bash
julia --project=. run.jl --config=benchmarks_cg.toml
```

For the 9-million-elements-per-GPU weak-scaling run:

```bash
julia --project=. run.jl --config=benchmarks_cg_multigpu.toml
```

Dagger currently contributes only its 1-GPU baseline because its CG vectors
are not distributed across processors yet.

Set solver controls per entry, e.g. `kwargs = { check_every = 10, max_iter = 1000 }`.
The problem is `tridiag(1,4,1) x = 1/2` from `x = 0`. Each solve checks convergence
(and syncs the residual to the host) only when `k % check_every == 0` or
`k == max_iter`, and fails if it does not converge (relative tolerance 1e-8 for
Float64, 1e-5 for Float32); use `max_iter = 1` for a single-update comparison.

`n_iter` counts complete solves per trial. Convergence determines the work, so
compare elapsed time — the CSV's GFLOP/s field is zero. Auto-sizing depends on
`max_iter`, so keep N fixed when comparing check intervals. The config pins
N=65,536 (set N=100,000,000 for the paper-sized workload); JACC additionally
requires N divisible by the GPU count. Validate the JACC partition kernels with
the selected GPUs visible:

```bash
julia --project=environments/jacc test/jacc_cg.jl
```
