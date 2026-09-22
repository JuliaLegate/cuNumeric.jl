# Reproduce the single-GPU CG benchmark

Use Julia 1.12 on a Linux GPU host with a working cuNumeric installation. From
this branch's repository root, create a separate environment (outside the repo):

```sh
julia --startup-file=no dev/iterativesolvers-smoke/setup_benchmark.jl "$HOME/cg-benchmark-env"
export BENCH_PROJECT="$HOME/cg-benchmark-env"
```

Setup develops **this checkout**, including the Krylov extension, and selects
Krylov 0.10.10, CUDA 6.4.0, and the Dagger commit used in the original runs.
Keep the resulting Manifest.toml with your results. If your working cuNumeric
installation needs LocalPreferences.toml library overrides, copy those settings
to this environment before running. Machine-specific library paths are not
checked in. Dubliner's existing configured environment can also be used directly:

```sh
export JULIA_DEPOT_PATH=/pool/emeitz/.julia:/home/emeitz/.julia
export BENCH_PROJECT=/pool/emeitz/iterativesolvers-smoke.wZrnP4
export JULIA=/home/emeitz/.julia/juliaup/julia-1.12.7+0.x64.linux.gnu/bin/julia
```

The existing dubliner environment contains the tested cuNumeric checkout and
extension. For testing subsequent branch edits, develop the updated checkout
into a separate environment with the setup script above.

## Run and change N

Each argument is a matrix dimension N. The launcher runs cuNumeric, CuArray,
and DaggerPatched sequentially, each size in a fresh Julia process:

```sh
# Start with defaults on H100.
unset CUBLAS_WORKSPACE_CONFIG
BENCH_ELTYPE=Float32 bash dev/iterativesolvers-smoke/run_benchmark.sh 8192 16384 32768 49152 65536
BENCH_ELTYPE=Float64 bash dev/iterativesolvers-smoke/run_benchmark.sh 8192 16384 32768 49152

# Reproduce the A30 workspace experiment, consistently across backends.
export CUBLAS_WORKSPACE_CONFIG=:32768:2
BENCH_ELTYPE=Float32 bash dev/iterativesolvers-smoke/run_benchmark.sh 65536

# One backend, any size:
BENCH_BACKENDS=cuNumeric bash dev/iterativesolvers-smoke/run_benchmark.sh 20000
```

`JULIA` selects the Julia executable. `BENCH_OUTPUT` selects a new output directory;
otherwise each invocation creates a timestamped directory. It contains a results
CSV, per-case logs, GPU information, package versions, Git revision, and workspace
configuration. Failed cases retain their logs; the launcher continues other cases
and exits nonzero. DaggerPatched explicitly patches Dagger's GPU matvec to call
`mul!` instead of CPU BLAS; use `BENCH_BACKENDS=Dagger` to reproduce stock behavior.

The launcher uses ordinary `NDArray(host_matrix)` construction, without a transpose
workaround or profiling. Two warm-ups precede five synchronized timed solves using
a reusable Krylov workspace. Setup and transfers are outside the timer. The dense
SPD matrix is generated from tridiagonal coefficients; every GPU multiplication
uses the full dense matrix. Every case checks convergence and an independent
Float64 residual (tolerance 1e-5 for FP32, 1e-8 for FP64).

The default Legate configuration matches dubliner: one GPU, 22000 MiB framebuffer,
65536 MiB system memory, 1024 MiB zero-copy memory. Override `LEGATE_CONFIG` for a
different machine or larger N. A 65536-square FP32 matrix alone occupies 16 GiB;
FP64 doubles this, and construction can require additional host/device copies.
Dagger's original largest cases failed allocating a second matrix on the 24 GiB
A30. H100 capacity and its default cuBLAS behavior can differ.

The workspace setting is an empirical A30 workaround, not a required H100 setting.
Use a consistent setting for a complete comparison and retain default results.
See [investigation and recorded measurements](gemv-investigation.md). Existing
CSVs and plots are historical A30 results; the workspace CSV distinguishes
profiled diagnostic measurements from unprofiled measurements.
