# GEMV performance investigation: ordinary user code

The user-facing target is `A = NDArray(host_A); Krylov.cg(A,b)` with no layout
choice, transpose workaround, or problem-size-specific behavior at the call site.

## Configuration controls

- `CUPYNUMERIC_MATMUL_CACHE_SIZE` is used by matrix-matrix tiling. The C++
  `contract` matrix-vector specialization directly submits MATVECMUL; it does
  not use that cache-size path.
- Minimum GPU chunk settings govern runtime partitioning, not cuBLAS kernel
  tile sizes. With one GPU they do not force a GEMV to split into many tiles.
- `CUPYNUMERIC_FAST_MATH` enables TF32 math mode. This changes numerical
  behavior and is not a like-for-like full-FP32 fix.
- `CUBLAS_WORKSPACE_CONFIG=:32768:2` provides two 32-MiB workspace blocks.
  Initial direct cuBLAS tests at N=65536, default math mode, reduced transposed
  SGEMV from ~119 ms to ~17 ms. Both 32-bit and 64-bit GEMV interfaces improve.
  `:65536:2` was also fast. Full CG verification is recorded below when complete.

Sources:
https://docs.nvidia.com/cupynumeric/26.06/api/settings.html
https://docs.nvidia.com/cuda/archive/13.1.0/cublas/index.html#cublassetworkspace
https://docs.nvidia.com/cuda/archive/12.4.0/cuda-toolkit-release-notes/

## Matrix construction and alternative internal layout

The current benchmark materializes `Matrix(SymTridiagonal(...))` before upload;
no SymTridiagonal reaches Krylov or the GPU. The same cliff also reproduced for
all-ones dense matrices, so it does not require this matrix's sparsity pattern.
Changing the matrix entries is not expected to fix the workspace-dependent
kernel selection.

An isolated constructor prototype preserves Julia column-major buffers through
an internal transformed store while retaining owned host memory. The caller
still uses `NDArray(host_A)`. This passed 130 ownership checks across all
supported element types (including empty/singleton shapes), successful-path
constructor inference checks, 20 conversion-lifetime tests, 29 permutation
tests, and 2675 vector-linalg tests. Further tests stopped because the smoke
environment lacks TensorOperations, not due to a numerical failure. This is
not a production-ready default layout change; no package source was modified.
The workspace solution is preferable if it fixes the unchanged implementation.

## Verified workspace solution

Original row-major construction, unchanged Krylov CG, default cuBLAS math mode:

```
CUBLAS_WORKSPACE_CONFIG=:32768:2
BENCH_LAYOUT=row BENCH_ELTYPE=Float32
```

At N=65536, five warmed solves had median 258.523151 ms, range
246.597447–268.508450 ms, 11 iterations, and independently checked relative
residual 7.761163694105887e-6 (required <=1e-5). The previous default-workspace
median was 1409.988826 ms. No prototype constructor was loaded for this run.

Independent direct-cuBLAS controls, all default math mode and unchanged input:
- Default workspace: transposed SGEMV ~118.97 ms.
- Explicit 64-MiB workspace via cublasSetWorkspace_v2: ~17.10–17.67 ms.
- Environment pool `:32768:2`: ~17.10–17.57 ms.
- Environment pool `:65536:2`: ~17.09–17.67 ms.

Thus this is a workspace-dependent kernel-selection fallback. Row-major
storage does not require the sevenfold penalty, and changing the input matrix
or asking users to transpose is unnecessary. TF32 also made the all-ones
microbenchmark faster (~20 ms), but is not the recommended solution: adequate
workspace fixes the case without changing arithmetic precision.

For reproducible paper data, apply the same workspace configuration to all
backends, record it as an execution setting, and keep timed input construction
and convergence criteria consistent. The earlier purple caller-side layout
workaround is diagnostic evidence only and is not the proposed user API.

For a production default, the backend can provision sufficient per-handle
workspace automatically. cuBLAS documents that cublasSetStream resets a custom
workspace, so that configuration must happen after each stream change. A Julia
package should not silently overwrite an explicitly chosen user workspace
configuration; the environment setting is currently a validated run-level fix,
not a committed production default.

Independent confirmation (Nsight attached, diagnostic rather than paper timing):
N=8192: 71.488 ms; N=32768: 125.886 ms; N=65536: 258.106 ms
(range 254.925–261.993 ms). All convergence/residual checks passed. The large
capture shows 11 `gemv2T_kernel_val` launches, median 17.500 ms each, plus
split-K reduction. The slow GEMM-named kernel is gone. The source still uses
ordinary `NDArray(host_A)` and default math mode.

CuArray with the SAME `:32768:2` workspace setting, unprofiled:
N=8192: 4.583 ms; N=32768: 55.911 ms; N=65536: 181.541 ms.
Thus the current fair largest-case comparison is 258.523 vs 181.541 ms,
not 258.523 vs the old default-workspace CUDA result. Dagger and the complete
size/precision sweep have not yet been rerun with this common configuration;
existing plots have intentionally not been patched with mixed settings.

The larger workspace restores the specialized transposed GEMV kernel. The
experiments establish workspace dependence; they do not identify the exact
minimum workspace or reverse-engineer cuBLAS's internal selection threshold.
