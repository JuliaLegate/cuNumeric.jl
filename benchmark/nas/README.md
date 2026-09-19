# NAS benchmarks

This directory documents the benchmark-independent NAS contract.
Backend implementations live under `src/<model>/benchmarks/nas/`.

The executable specification is GMAP/NPB-GPU at commit
`3f12d84920ee315ab00ef283717c1e74b68f4d00`. Do not update the commit without
updating reference metadata and validating the official verification values.
Its license is preserved in [`THIRD_PARTY_LICENSE.md`](THIRD_PARTY_LICENSE.md).

## Comparing models fairly

These compare implementations of the same mathematical problems, not identical
kernels or certified NPB scores. Use the same class, Float64, iteration count,
visible GPUs, and warmup policy, and require official verification to pass.
Do not interpret throughput differences as runtime overhead alone.

| Benchmark | Shared work | Material differences still included in results |
| --- | --- | --- |
| EP | Exact RNG sequence, MK=8, Gaussian transform, histogram/sum partials | cuNumeric/cuPyNumeric traverse arrays at each recurrence step and evaluate masked rejected-pair math; CUDA/JACC/Dagger keep each stream local. Global aggregation is untimed for all. |
| FT | Exact initial field, forward FFT, fixed evolve/inverse/checksum iterations | Host RNG in cuNumeric/cuPyNumeric/Dagger versus device RNG in CUDA/JACC; native FFT implementations and checksum strategies differ. Dagger leaves global checksum aggregation untimed. |
| MG | Exact RHS, hierarchy, operators, fixed V-cycles and L2 verification | Direct kernels versus separable transfers/temporaries; Dagger restriction computes extra fine-grid stencil outputs and leaves global norm aggregation untimed. |

For multi-GPU runs, distinguish **can execute with multiple GPUs visible** from
**the dominant operation is distributed**. EP partitions independent streams;
MG's array adapters can partition levels (coarse levels expose less parallelism).
Dagger FT uses a distributed FFT, but the current cuNumeric/cuPyNumeric full 3-D
FFT is unpartitioned. CUDA is the single-GPU baseline; JACC FT and MG are
single-GPU adapters. No multi-GPU scaling claim follows from correctness alone.

Each adapter's header records its limitations. These include choices in the
current adapter, not only fundamental restrictions of the programming model.

## EP execution contract

One timed sample generates the class's exact `2^(M+1)` random numbers with the
NPB 46-bit linear-congruential generator, applies the Gaussian
acceptance-rejection transform, and produces the ten-bin histogram plus `sx`
and `sy` partial sums. Correctness aggregation happens after timing and checks
the official sums with relative tolerance `1.0e-8`.

NPB explicitly permits changing `MK`, the batch-size exponent, without changing
the result. The pinned CUDA source uses `MK=16`; all harness models use `MK=8`.
That common setting preserves the exact global RNG sequence while limiting each
independent stream to 256 Gaussian pairs, allowing array programming models to
express the serial recurrence without host-generating the benchmark workload.
Skip-ahead masks/constants and output reset are setup; random samples and
Gaussian transforms remain timed. The reference also aggregates partials after
its kernel timer, but uses a different partial/block layout.

cuNumeric and cuPyNumeric implement the LCG as Float64 array algebra because
they have no NPB RNG primitive. CUDA.jl and JACC evaluate the same scalar stream
function directly. Dagger maps that function over its device-resident chunks
without a hand-written CUDA kernel. JACC and Dagger partition streams across
the requested GPUs; CUDA.jl remains the single-GPU baseline.

Use `n_iter = 1`; use `n_trial` for independent complete runs. The common
throughput value follows NAS EP and counts random numbers generated rather than
floating-point instructions.

Run class S across all models with:

```sh
julia --project=. run.jl --config=benchmarks_nas_ep.toml
```

## FT execution contract

FT needs no cached input artifact: its 46-bit RNG and spectral index map are
part of the benchmark. One timed sample follows the pinned `CUDA/FT/ft.cu`:

1. Generate the class's exact complex initial field and exponential index map.
2. Perform one forward 3-D FFT.
3. For each official `NITER`, evolve the spectrum cumulatively, perform an
   inverse 3-D FFT, and compute the prescribed 1024-point checksum.
4. Fetch checksums only after the run and compare every iteration with the
   official values at relative tolerance `1.0e-12`.

Use `n_iter = 1`; use `n_trial` for independent complete runs. CUDA.jl uses
device kernels plus cuFFT. JACC uses JACC kernels but must call cuFFT because it
has no FFT API. Dagger uses its distributed 3-D FFT. cuNumeric and cuPyNumeric
use native Legate FFT auto tasks, whose constraints broadcast all transformed
axes. Their full 3-D transform therefore has no partitionable batch axis and
is not a distributed FFT. See `src/ndarray/detail/fft.jl` in the parent package
and cuPyNumeric's `DeferredArray.fft`.

CUDA/JACC gather 1024 samples (JACC reduces real/imaginary parts separately);
cuPyNumeric uses native `take`. cuNumeric and Dagger currently scan a full-volume
mask, a significant extra cost. Dagger only reduces within each slab during
timing, so it also omits the cross-GPU aggregation paid by other adapters.
All adapters use normalized inverse FFTs instead of NPB's unnormalized inverse
and checksum-only scaling. Host RNG, transfers, FFT scratch/temporary allocation
and planning performed inside `run` remain charged to that model. In particular,
cuPyNumeric's serial Python RNG should not be mistaken for FFT runtime cost.

Run class S across all models with:

```sh
julia --project=. run.jl --config=benchmarks_nas_ft.toml
```

## MG execution contract

One timed MG sample implements the V-cycle from pinned `CUDA/MG/mg.cu`: starting from the
official sparse right-hand side, it computes the initial residual, executes
the class's fixed number of complete multigrid V-cycles, recomputes the finest
residual after every cycle, and performs the reference's initial and final L2
norm reductions. Reduction results remain device-side until correctness
verification. The 46-bit RNG search for the ten positive and ten negative
impulses is setup work outside NPB-GPU's timer and is likewise performed before
harness timing.

Timing is not identical to standalone NPB: every adapter clears the solution
hierarchy inside each sample, computes initial/final L2 sum-of-squares, and
defers square root/normalization and verification to the untimed check. NPB's
additional Linf reduction is omitted by all adapters. Dagger computes only
per-slab sum-of-squares during timing and combines these after timing; this
remaining asymmetry matters particularly for small grids and multi-GPU results.

The V-cycle includes periodic boundary exchange, the 27-point residual,
full-weight restriction, trilinear interpolation, and the NPB smoother at
every prescribed level. Correctness compares the final L2 norm with the
official value at relative tolerance `1.0e-8`.

JACC is single-GPU because it has no distributed 3-D halo API. CUDA.jl is the
single-GPU baseline. cuNumeric, cuPyNumeric, and Dagger express the hierarchy
through their distributed array APIs; implementation headers document their
communication limitations.

Run class S across all models with:

```sh
julia --project=. run.jl --config=benchmarks_nas_mg.toml
```
