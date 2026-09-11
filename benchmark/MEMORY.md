# Memory accounting and supported runs

`src/memory.jl` is the authoritative preflight model. The older `total_space`
helpers describe storage and are not used by the sweep planner. Byte arithmetic
uses `BigInt`; GPU count, backend, dtype, fusion, benchmark type, dimensions,
and warmup/iteration count all enter the planning calculation.

## Lifetime policy

Estimates are conservative upper bounds, not measured allocator peaks. They
include initialization and the entire trial. Julia tracing GC is not assumed
to run between iterations. In the baseline Monte Carlo kernel an unreferenced
broadcast output can therefore remain for every iteration. Gray-Scott baseline,
begin and expression forms can similarly retain named buffers until GC. The
function/let acceleration forms explicitly destroy last-used local arrays and
are bounded independently of iteration count. Fusion-disabled execution still
uses the accelerated lifetime rewrite.

Fused Gray-Scott is bounded by four persistent grids plus four named interior
results and an assignment output. Unfused execution reserves three additional
interior buffers for nested operands and the outer broadcast destination.
The function/let bound conservatively retains named intermediates within an
iteration even when inter-statement fusion may eliminate them. It does not
claim an exact optimized peak. Full parent grids are counted rather than
assuming slice/halo partition placement.

Python reference counting avoids the Julia retention allowance, but an input
and output of an unfused operation must coexist. The random helper generates
Float64 values before casting, and this conversion is counted. These models
assume the current kernels and standard supported runtime execution; physical
Legate instance lifetimes still require validation on the target machine.

## Native workspace

GEMM, DMD, FFT and tensor contraction scratch/packing bounds depend on native
libraries and their algorithms. The previous arbitrary extra-array allowances
are not treated as verified bounds. Supply a verified **per-GPU byte bound**:

```toml
[workspace.gemm]
cunumeric = 268435456
cudajl = 268435456
cupynumeric = 268435456
```

The numbers above demonstrate syntax only, not recommended bounds. Use a bound
verified for the maximum dimensions in the sweep and installed library versions.
Other keys are the registered names, e.g. `workspace.dmd_baseline` and
`workspace.tensor_contract4`. Each enabled native backend needs an entry;
missing entries fail preflight, including for pinned dimensions. Zero is valid
only when no extra workspace/packing is required. No calibration, OOM retry,
or automatic problem reduction is performed during execution.

DMD counts the entire SVD on one GPU regardless of P. Its shared baseline is
limited by the largest requested GPU count's scaled problem. This is a memory
guard, not a distributed SVD implementation. Full factors/parent stores and
complex outputs are included in addition to the supplied workspace bound.

GEMM and contractions conservatively count full operands on each GPU until
mapper replication bounds are verified. Poisson partitions batches and retains
the inverse Laplacian per GPU; Python FFT outputs are conservatively treated
as complex128. Native workspace remains separate.

## Budgets and validation

`mem_frac` applies to the smallest visible GPU capacity, capped by
`CUNUMERIC_BENCH_FBMEM_MB` when provided. That cap is also passed to Legate.
`CUDA_VISIBLE_DEVICES` numeric indices and GPU UUIDs are resolved explicitly;
unresolvable/MIG identities are rejected rather than guessed. Free memory must
cover the budget before starting. External GPU users can invalidate that check.

Use `--dry-run` to see initialization, iteration, workspace and peak bytes for
each configuration. CPU tests verify dispatch and planner invariants; they do
not verify native allocator bounds. Before calling a configuration GPU-validated,
check its runtime allocation trace and execute the requested GPU sweep with the
installed native libraries. Record the verified workspace bounds in its config.

```bash
julia --project=. test/runtests.jl
```
