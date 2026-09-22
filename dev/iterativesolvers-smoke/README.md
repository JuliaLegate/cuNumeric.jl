Run the benchmark: [setup, configurable N, and workspace settings](BENCHMARK.md).

Latest GEMV finding (2026-09-22): increasing cuBLAS workspace fixes the largest FP32 cliff with ordinary row-major construction and unchanged CG. See [GEMV investigation](gemv-investigation.md). The transposed-storage plots below are earlier diagnostics, not the proposed paper configuration.

# Stock IterativeSolvers smoke test

`cg.jl` calls stock CG with NDArrays inside `@allowautofetch` and checks the
returned solution against a host-computed residual. `other_solvers.jl` repeats
CG and tries three more solvers, printing their errors without adapting them.
The baseline results below were collected without modifying either library.

Tested on dubliner's GPU with Julia 1.12.7, IterativeSolvers 0.9.4, and
`codex/ndscalar-autounwrap` at `e0e2f7fd2c962c3aaf48454d7701bd2c3189c5dd`.
The input is a 32-by-32 Float64 SPD tridiagonal matrix with an all-ones RHS.

| Solver | Observed result |
| --- | --- |
| CG | Pass; relative residual `2.2633986948989007e-16`. |
| MINRES | `MethodError` constructing `MINRESIterable`: work vectors must be `DenseVector`; its two small vectors also have incompatible types (`Vector{Float64}` versus `Vector{CNFloat{Float64}}`). |
| GMRES | Scalar-indexing error in `copyto!(first_col, b)` at `gmres.jl:241`: the basis is a host `Matrix`. |
| BiCGStab(l) | Scalar-indexing error in `copyto!(residual, b)` at `bicgstabl.jl:47`: the workspace is a host `Matrix`. |

## Run on dubliner

The fresh environment and full logs are in
`/pool/emeitz/iterativesolvers-smoke.wZrnP4` (`cg.log`, `other_solvers.log`).

```sh
cd /pool/emeitz/iterativesolvers-smoke.wZrnP4
export JULIA_DEPOT_PATH=/pool/emeitz/.julia:/home/emeitz/.julia
export LEGATE_AUTO_CONFIG=0
export LEGATE_CONFIG='--gpus 1 --cpus 2 --fbmem 2048 --sysmem 2048 --zcmem 256'
/home/emeitz/.julia/juliaup/julia-1.12.7+0.x64.linux.gnu/bin/julia --startup-file=no --project=. cg.jl
# Replace cg.jl with other_solvers.jl to reproduce the other failures.
```

The supplied Project.toml selects the scalar Git branch through `[sources]`.
On this machine, Pkg's URL checkout failed with
`GitError(Code:ERROR, Class:Submodule, cannot get submodules without a working tree)`.
The tested environment instead uses `cuNumeric = {path = "cuNumeric.jl"}` pointing
to a fresh Git clone of that exact branch and revision. Its backend
library preferences were copied from the working dubliner test environment.
That clone now also contains the Krylov extension under development on this
experimental branch; the baseline logs remain unchanged.

## Krylov and KrylovKit

Run `krylov.jl` in the same environment (log: `krylov.log`). Tested releases:
Krylov 0.10.10, KrylovKit 0.10.4, VectorInterface 0.6.1, CUDACore 6.4.0.
The script compares NDArrays with CUDA CuArrays on the same SPD system, with
scalar indexing disabled and `@allowautofetch` enabled. The following baseline
results predate the Krylov extension.

| Call | NDArray | CuArray |
| --- | --- | --- |
| `Krylov.cg(A, b)` | Missing `NDArray{Float64,1,Nothing}(undef, n)` constructor at `krylov_workspaces.jl:273`. | Pass, residual `2.23e-16`. |
| `Krylov.cg!` with `CgWorkspace(KrylovConstructor(b))` | Gets past allocation; fails at `cg.jl:239` because `kaxpy!` requires matching coefficient and vector element types (`CNFloat{Float64}` versus `Float64`). | Pass, residual `2.23e-16`. |
| `KrylovKit.linsolve(A, b, zero(b), CG(...))` | Fails at `linsolve/cg.jl:4`: VectorInterface's `inner` calls two-input `mapreduce`, which cuNumeric does not support. | Pass, residual `2.39e-16`. |

Both libraries explicitly support CUDA arrays:
[Krylov GPU support](https://jso.dev/Krylov.jl/stable/gpu/) and
[KrylovKit vector support](https://jutho.github.io/KrylovKit.jl/stable/).
Krylov's documented custom workspace bypasses the constructor gap but does not
resolve its scalar dispatch requirements. KrylovKit's first integration point
is `VectorInterface.inner`; later compatibility has not been established.

## Krylov extension

This branch adds `cuNumericKrylovExt`. It forwards `kaxpy!` and `kaxpby!` for
`DeviceScalar` coefficients to existing LinearAlgebra methods. Following
[Dagger's workspace integration](https://github.com/JuliaParallel/Dagger.jl/blob/master/ext/KrylovExt.jl),
`CgWorkspace(A, b::NDArray)` allocates through `KrylovConstructor` and `similar`.
Thus direct `@allowautofetch Krylov.cg(A, b)` no longer needs an `undef`
constructor. Complex CG additionally needs `@allowpromotion` for its real
coefficients to scale complex vectors.

The helper extension points are documented in
[Krylov's custom-vector guide](https://jso.dev/Krylov.jl/stable/custom_workspaces/#Methods-to-overload-for-compatibility-with-Krylov.jl).
Focused tests live in `test/array/krylov.jl`; dubliner's results are in
`krylov-extension.log`. To test this extension in a new environment, develop
this branch's checkout instead of selecting the scalar baseline in `[sources]`.
Other solver workspace constructors and KrylovKit support are not added here.

## Single-GPU performance comparison

`benchmark_cg.jl` compares cuNumeric, Dagger, and plain CuArray with Krylov CG.
Run each backend in a separate process, sequentially on the same GPU:

```sh
julia -t4 --startup-file=no --project=. benchmark_cg.jl Dagger
julia -t4 --startup-file=no --project=. benchmark_cg.jl cuNumeric
julia -t4 --startup-file=no --project=. benchmark_cg.jl CuArray
```

The benchmark uses Float64 dense SPD tridiagonal matrices (stored densely),
sizes 256, 1024, and 4096, relative tolerance `1e-8`, and no preconditioner.
Dagger uses one GPU-resident chunk per array; its chunk types are asserted.
Two warm-up solves precede five timed solves reusing one workspace. Timers
include backend synchronization but exclude allocation, compilation, host
transfers, residual validation, and explicit GC between samples. Each result
reports iterations, median/min/max milliseconds, relative residual, and all
five samples. These measurements describe this CG workload, not general
backend throughput. Optional sizes may follow the backend argument.

### Results on dubliner (2026-09-21)

One NVIDIA A30X (24 GiB), Julia 1.12.7 with four Julia threads, Krylov 0.10.10,
CUDA 6.4.0, and Dagger commit `ee7fcb13a21252878d738290c85af57de2824b7b`.
cuNumeric uses the scalar baseline plus this branch's Krylov extension;
broadcast fusion uses the default enabled preference. Legate options are the
same as the smoke-test command above (one GPU, two CPU processors).

**Unmodified Dagger failed before completing CG.** Its `matvecmul!` at
`src/array/mul.jl:522` calls CPU `BLAS.gemv!` on CuArrays and errors with
`unsafe_convert(::Type{Ptr{Float64}}, ::CuPtr{Float64})`.
`DaggerPatched` is a separately selectable benchmark mode that adds one
CuArray-specific `Dagger.matvecmul!` method forwarding to `mul!`. No files in
the Dagger checkout were edited. The Dagger timings below require this workaround.

| Dense matrix size | CG iterations | cuNumeric median (ms) | DaggerPatched median (ms) | CuArray median (ms) |
| --- | ---: | ---: | ---: | ---: |
| 256 × 256 | 18 | 162.532 | 77.106 | 2.573 |
| 1024 × 1024 | 19 | 124.932 | 81.225 | 2.746 |
| 4096 × 4096 | 19 | 125.229 | 80.428 | 5.239 |

All relative residuals were between `5.25e-9` and `6.09e-9`. At size 4096,
cuNumeric took 1.56 times DaggerPatched's time and 23.9 times CuArray's time.
The nearly flat scheduler-backed timings suggest fixed per-iteration overhead
dominates this workload, but these timings do not identify the cause without
profiling. The 256 cuNumeric samples varied from 118 to 173 ms; see the raw
samples in `benchmark-results.csv` rather than treating the medians as precise
backend constants.

The environment and logs remain in `/pool/emeitz/iterativesolvers-smoke.wZrnP4`.
For the working Dagger comparison, run `benchmark_cg.jl DaggerPatched`; the
`Dagger` mode deliberately reproduces the unmodified failure.

### Larger matrices

The same benchmark was extended to 8192, 16384, and 32768, with each
backend/size pair in its own process. Two warm-ups, five samples, tolerances,
thread counts, and the single-chunk Dagger layout are unchanged. To fit the
8 GiB matrix in GPU memory, these runs used:

```sh
export LEGATE_CONFIG='--gpus 1 --cpus 2 --fbmem 20000 --sysmem 32768 --zcmem 1024'
julia -t4 --startup-file=no --project=. benchmark_cg.jl cuNumeric 32768
# Repeat sequentially for DaggerPatched and CuArray, and for 8192 and 16384.
```

| Dense matrix size | Matrix storage | cuNumeric median (ms) | DaggerPatched median (ms) | CuArray median (ms) |
| --- | ---: | ---: | ---: | ---: |
| 8192 × 8192 | 512 MiB | 122.384 | 96.284 | 12.679 |
| 16384 × 16384 | 2 GiB | 235.207 | 142.287 | 42.462 |
| 32768 × 32768 | 8 GiB | 271.726 | 272.748 | 159.237 |

Every solve used 19 iterations, and all relative residuals were below `5e-9`.
At 32768, cuNumeric and DaggerPatched are effectively tied: their sample ranges
overlap (269.6–275.3 ms and 250.3–275.4 ms, respectively). Both remain about
1.7 times CuArray's time. These data show the smaller-matrix gap closing, not
a statistically established cuNumeric lead. Logs are named
`benchmark-large-<backend>-<size>.log` in the same remote environment; the CSV
includes every sample.

### Precision sweep

Set `BENCH_ELTYPE=Float32` to use Float32 arrays and a relative tolerance of
`1e-5` (the default remains Float64 with `1e-8`). Residuals are checked in
Float64 against the exact stored tridiagonal coefficients, outside timing.
The GPU matrix remains dense. Output rows now include the element type.
Float32 and Float64 may converge in different iteration counts because their
tolerances differ; compare backends within each precision.

The largest runs use a 22,000 MiB framebuffer allowance:

```sh
export LEGATE_CONFIG='--gpus 1 --cpus 2 --fbmem 22000 --sysmem 65536 --zcmem 1024'
BENCH_ELTYPE=Float64 julia -t4 --startup-file=no --project=. benchmark_cg.jl cuNumeric 49152
BENCH_ELTYPE=Float32 julia -t4 --startup-file=no --project=. benchmark_cg.jl cuNumeric 65536
```

Repeat with `DaggerPatched` and `CuArray`; the Float32 sweep covers 8192,
16384, 32768, 49152, and 65536. Every backend/size pair runs sequentially in
a fresh process, with two warm-ups and five measured solves. Logs are named
`benchmark-precision-<type>-<backend>-<size>.log` in the remote environment.

For **49152² Float64 (18 GiB)**, cuNumeric's median was **536.207 ms** versus
**446.768 ms** for CuArray: 1.20 times the CuArray time. Both took 18 iterations
and achieved a relative residual of `9.498e-9`.
DaggerPatched failed during setup: the concise-error retry confirmed
`OutOfGPUMemoryError` attempting an 18 GiB allocation with 18 GiB already
in use by its memory pool. Its retry log ends in `49152-retry.log`.

Float32 results (milliseconds per solve):

| Matrix size | Matrix storage | Iterations | cuNumeric | DaggerPatched | CuArray |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8192² | 256 MiB | 11 | 68.471 | 56.218 | 4.560 |
| 16384² | 1 GiB | 11 | 79.341 | 75.960 | 19.520 |
| 32768² | 4 GiB | 12 | 124.682 | 135.667 | 55.937 |
| 49152² | 9 GiB | 11 | 215.952 | 167.454 | 111.981 |
| 65536² | 16 GiB | 11 | 1409.989 | GPU allocation failed | 193.286 |

All successful solves passed independent Float64 residual checks at the
requested tolerance. Dagger's 65536² Float32 setup failed with
`OutOfGPUMemoryError` while trying to allocate another 16 GiB; the error
reported 16 GiB already in use by its pool. This is a failure of the tested
single-chunk construction path, not a claim that every Dagger layout needs
that much memory.

cuNumeric did **not** catch CuArray in these measurements. At 65536² Float32,
its time jumps to 1.41 seconds (sample range 1.407–1.459 seconds), 7.3 times
CuArray's time. This is a repeatable cliff across the five samples despite
the same 11 iterations as 49152²; it needs profiling before assigning a cause.
Dagger's 16384² Float32 samples also varied substantially (54–112 ms), so its
76 ms median should not be overinterpreted.

Raw samples are in `benchmark-precision-results.csv`. The benchmark catches
Dagger errors and prints only their root exceptions: the default task error
printer attempted to stringify the huge matrix in the original failed
Float64 run, which was stopped after about five minutes.

### Profiling the Float32 65536² cliff

An Nsight Systems 2025.3.2 capture of one warmed solve identified the immediate
bottleneck: 11 `ampere_sgemm_128x128_tn` launches totalled **1.374 seconds**,
or **99.9% of GPU kernel time** (median 118.8 ms per launch). All recorded
memory copies together took 0.719 ms: 0.525 MB device-to-device and only tiny
host/device copies. There were no matrix-sized transfers during the capture.
This identifies a slow GEMM-named GPU kernel, rather than matrix spilling,
as the dominant cost of this warmed solve. A subsequent API-level trace
(below) confirmed that cuBLAS launches this kernel from GEMV. A GPU kernel
name alone does not identify the public library entry point.

The Julia matvec uses `contract`, whose matrix-vector specialization launches
the backend MATVECMUL task and calls cuBLAS GEMV. This case does not take the
general cuTENSOR contraction path. `@accelerate` does not change this kernel selection and cannot
rewrite inside the opaque `Krylov.cg!` call. It currently rejects control flow
and can only optimize individual straight-line function bodies/blocks.

The textual summary is in `profile-fp32-65536-stats.txt`; full `.nsys-rep` and
SQLite traces remain in the remote experiment environment. Reproduce with:

```sh
mkdir -p nsys-tmp
export TMPDIR="$PWD/nsys-tmp"
# Use the precision-sweep Legate configuration and Julia environment above.
BENCH_ELTYPE=Float32 BENCH_PROFILE=true nsys profile --trace=cuda \
  --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi \
  --capture-range-end=stop -o cg-fp32-65536 \
  julia -t4 --startup-file=no --project=. benchmark_cg.jl cuNumeric 65536
```

`BENCH_PROFILE=true` captures the first measured solve after two warm-ups.
Profiled timings are diagnostic and are not substituted into the performance
CSV or plots.

A matching CuArray capture at Float32 65536² recorded 11
`gemv2N_kernel<...>` launches totalling **190.487 ms** (17.317 ms average),
plus 0.293 ms in 11 split-K reduction launches. The installed cuBLAS.jl
`mul!` method calls `gemv!`, which selects `cublasSgemv_v2_64` for cuBLAS
12 or newer. NDArray's `mul!` also reaches GEMV through the contraction
specialization, but ran `ampere_sgemm_128x128_tn`. The common GEMV operation
does not imply the same GPU kernel for different arguments/layouts.

The `128x128` suffix describes an internal GEMM tile, not cuNumeric issuing
separate calls for every 128-by-128 piece. The trace contains one main GEMM
launch per CG iteration. The corresponding CuArray summary is saved in
`profile-cuarray-fp32-65536-stats.txt`; its full remote capture is named
`cg-cuarray-fp32-65536.nsys-rep`. Capture it with the same command as above,
replacing the output name and backend argument with the CuArray versions.

**API-level correction:** re-profiling with `--trace=cuda,cublas-verbose`
recorded 11 `cublasSgemv_v2` calls and no GEMM API calls from the cuNumeric
backend. The GEMM-named kernel is selected internally by cuBLAS. The initial
claim that the general cuTENSOR path caused this kernel selection was wrong.
cuNumeric already exposes GEMV through `contract`; no new wrapper entry point
is needed merely to reach GEMV. Relevant differences to investigate next are
row-major versus column-major storage/transpose mode and the 32-bit versus
64-bit cuBLAS entry points. Their causal contribution has not yet been isolated.
Evidence: `profile-cunumeric-blas-fp32-65536-stats.txt`; full remote capture:
`cg-cunumeric-blas-fp32-65536.nsys-rep`.

### Controlled cuBLAS layout comparison

`diagnose_gemv.jl` allocates one Float32 CuArray and compares the 32-bit and
64-bit GEMV interfaces with both transpose modes. Each case has three warm-ups
and five synchronized timings, and checks the output. Installed cuBLAS: 13.7.0.
These measurements exclude cuNumeric and Legate entirely.

| N | 32-bit N (ms) | 32-bit T (ms) | 64-bit N (ms) | 64-bit T (ms) |
|---:|---:|---:|---:|---:|
| 49152 | 11.167 | 13.210 | 10.075 | 13.229 |
| 65535 | 21.279 | 118.951 | 18.482 | 118.982 |
| 65536 | 19.082 | 118.943 | 17.428 | 118.971 |

The transpose-mode path reproduces the cliff independently of cuNumeric.
Changing the integer interface does not remove it. This is not restricted to
exactly N=65536. The previous trace identifies the expensive kernel selected
inside GEMV; these controlled timings isolate the relevant API argument.

For the full cuNumeric solve, `BENCH_LAYOUT=column` is an experimental option
that attaches the Julia matrix using Legate's existing column-major attachment
API. It changes input construction only, not `mul!`, the wrapper, or Krylov.
Whether a layout survives GPU mapping must be checked by measurement rather
than inferred from its host attachment.

The column-major attachment experiment did **not** remove the cliff: warmed
CG median **1363.022 ms** (range 1362.067–1375.410 ms), 11 iterations,
Float64-checked relative residual **7.76171e-6**. Initial warm-up also took
several minutes. Host attachment order alone is therefore not a fix for the
GPU operation. The backend source's MATVECMUL mapper requests default exact
layouts; preserving/choosing a GPU column-major matrix instance is a separate
backend concern. No production NDArray construction or wrapper was changed.

The next implementation target is the backend's physical layout / GEMV kernel
selection, not adding a GEMV wrapper or changing the Julia `contract` call.
A backend layout change must account for the one-time matrix conversion cost
and avoid converting the matrix on every multiply. These experiments establish
the cause and a reproducer; they do not yet provide a validated production fix.

### Public issue research (2026-09-21)

Searched NVIDIA cuBLAS release notes, NVIDIA developer forum reports, and
cuPyNumeric release notes/issues for transposed SGEMV, large dimensions,
65535/65536, and the observed kernel name. No exact public match was found
for this Ampere Float32 timing cliff. This is not evidence that NVIDIA has
no internal report.

Relevant but nonmatching reports:
- https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
  CUDA 13.4 describes transposed GEMV index-range correctness fixes; CUDA
  13.3/13.3 Update 1 lists shape/workspace failures and Blackwell correctness
  issues. None documents this Ampere performance cliff.
- https://github.com/nv-legate/cupynumeric/releases
  cuPyNumeric 26.06 notes einsum performance regressions on Blackwell starting
  with cuBLAS 13.2. Different architecture and operation; not confirmation.
- https://forums.developer.nvidia.com/t/fast-gemv/15396
  A 2010 first-hand report demonstrates poor transposed SGEMV performance
  on much older hardware. Historical precedent only, not this bug.

Python reproduction uses the existing isolated conda environment
`/home/emeitz/miniconda3/envs/cupynumeric-bench-26.6`, cuPyNumeric 26.06.01,
Legate 26.06.01, and cuBLAS 13.4.1.3 (the direct Julia experiment used 13.7.0).
`diagnose_cupynumeric.py` verifies its output and records loaded cuBLAS paths.

Python reproduction confirmed the same kernel-selection cliff using native
`cn.ones` inputs (no NumPy attachment). With `LEGATE_AUTO_CONFIG=0` and
`LEGATE_CONFIG='--gpus 1 --cpus 2 --fbmem 22000 --sysmem 65536 --zcmem 1024'`,
run the conda environment's `python diagnose_cupynumeric.py N`.
Use `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` as in the experiment.

Nsight captures include three warm-up and five measured GEMVs:

| N | Python call median (ms) | Main GPU kernel median (ms) | Kernel |
|---:|---:|---:|---|
| 49152 | 13.976 | 13.044 | gemv2T_kernel_val |
| 65535 | 123.223 | 118.836 | ampere_sgemm_128x128_tn |

Outputs were checked exactly against N for the all-ones matrix and vector.
Only the output vector was copied to host (0.197/0.262 MB); no matrix transfers
appear in either capture. These are profiled diagnostic timings, not additions
to the CG benchmark CSV. The cliff therefore reproduces through Python
cuPyNumeric with cuBLAS 13.4.1.3 as well as direct cuBLAS 13.7.0 from Julia.

The exact N=65536 native allocation hits SIGFPE inside Realm in this Python
environment, before a usable GEMV measurement. N=65535 avoids this separate
failure and matches a size already verified slow in the direct cuBLAS test.
Initial NumPy-attached variants had expensive setup/noisy multi-second results;
they are not used as evidence for the GPU cliff. The final script uses native
cuPyNumeric inputs. No production library or wrapper code was changed.

### Column-major attachment: actual BLAS arguments

`diagnose_layout.jl` checks a nonsymmetric 1536-by-1024 Float32 matrix with
both ordinary row-major construction and the same column-major attachment
used in the earlier CG experiment. Both outputs agree with host multiplication.
With `CUBLAS_LOGINFO_DBG=1 CUBLAS_LOGDEST_DBG=stdout`, BOTH cases log:

```
cublasSgemv_v2: trans=CUBLAS_OP_T, m=1024, n=1536, lda=1024
```

Thus the column-major HOST attachment does not survive as column-major storage
at the GEMV boundary in this test. The earlier failed attachment workaround
was not a test of column-major GPU GEMV. The MATVECMUL mapper requests default
exact layouts; preserving/choosing the GPU instance layout remains the relevant
issue. `diagnose-layout.log` records the actual calls. This rectangular test
establishes the behavior directly; it does not log the earlier 65536 CG run.

### Lazy transposed storage workaround

A rectangular nonsymmetric correctness test now verifies:

```julia
A = permutedims(NDArray(permutedims(host)))
mul!(y, A, x)
```

The first permutation prepares storage for the transpose; the backend
`permutedims` returns a logical transpose view, restoring the original matrix.
The solver receives exactly the original operator, not its transpose. This
works for nonsymmetric matrices and does not rely on CG's symmetric test data.
The actual cuBLAS log changes to `trans=N, m=1536, n=1024, lda=1536`.
Both ordinary construction and direct column-major attachment instead produce
`trans=T, m=1024, n=1536, lda=1024`. No wrapper changes are needed.

`BENCH_LAYOUT=transposed_storage` enables this benchmark-only construction.
Setup is outside the timed solve, as with the original benchmark; each iteration
still calls the existing `mul!` and does not materialize another transpose.

The consistent transposed-storage FP32 sweep produced median solve times of
69.779, 87.987, 147.715, 180.934, and 278.705 ms for N=8192, 16384, 32768,
49152, and 65536. All five sizes passed the same Float64-checked residual
criterion, with unchanged iteration counts. The largest case improves from
1409.989 ms to 278.705 ms (~5.06x); the recorded CuArray comparison is 193.286 ms.
This layout is not faster at every smaller size. Raw samples are preserved in
`benchmark-layout-results.csv`; `plot_benchmarks.py --layout` generates
`cg-scaling-layout.{png,svg,pdf}` with both original and transposed-storage
FP32 series. FP64 was not rerun with this layout. This remains a benchmark
construction option, not a change to the default NDArray constructor.
