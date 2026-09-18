# NAS benchmarks

This directory documents the benchmark-independent NAS contract.
Backend implementations live under `src/<model>/benchmarks/nas/`.

The executable specification is GMAP/NPB-GPU at commit
`3f12d84920ee315ab00ef283717c1e74b68f4d00`. Do not update the commit without
updating reference metadata and validating the official verification values.
Its license is preserved in [`THIRD_PARTY_LICENSE.md`](THIRD_PARTY_LICENSE.md).

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
use native Legate auto tasks for the 3-D FFT, allowing runtime distribution.
Models without the NPB RNG or indexed reduction host-stage initialization
and/or use a masked reduction; each implementation header gives the exact
limitation.

Run class S across all models with:

```sh
julia --project=. run.jl --config=benchmarks_nas_ft.toml
```
