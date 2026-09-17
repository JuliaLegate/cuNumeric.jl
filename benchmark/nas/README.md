# NAS benchmarks

This directory documents the benchmark-independent NAS contract.
Backend implementations live under `src/<model>/benchmarks/nas/`.

The executable specification is GMAP/NPB-GPU at commit
`3f12d84920ee315ab00ef283717c1e74b68f4d00`. Do not update the commit without
updating reference metadata and validating the official verification values.
Its license is preserved in [`THIRD_PARTY_LICENSE.md`](THIRD_PARTY_LICENSE.md).

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
