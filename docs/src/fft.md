# FFT

GPU-only FFTs on `NDArray`, following
[AbstractFFTs.jl](https://github.com/JuliaMath/AbstractFFTs.jl) conventions:

- Every dimension is transformed unless `dims` is given.
- `fft` and `bfft!` are unnormalized. `ifft` divides by the transformed length.
- Real input is promoted to complex (`Float32` → `ComplexF32`, else `ComplexF64`).

| Function | Result |
|:---|:---|
| `fft(A, [dims])`, `ifft(A, [dims])` | new array |
| `fft!(A, [dims])`, `ifft!(A, [dims])`, `bfft!(A, [dims])` | overwrites `A` |
| `bfft!(dest, src)` | unnormalized inverse of `src` into `dest` |

```julia
using cuNumeric

A = cuNumeric.rand(ComplexF32, 64, 64)
Y = fft(A)       # all dimensions
Z = fft(A, 1)    # first dimension only
ifft(Y)          # ≈ A

fft!(A)          # in place
bfft!(A)         # ≈ length(A) * original A
```

In-place functions require a `ComplexF32` or `ComplexF64` array.

## Batched transforms

`batched_fft`, `batched_ifft` and their `!` forms treat dimension 1 as the
batch and transform the rest. The batch axis can split across GPUs. An all-axes
`fft(A)` cannot.

```julia
signals = cuNumeric.rand(ComplexF32, 32, 1024)   # 32 traces
batched_fft(signals)                             # FFT along dim 2
```

## Limitations

- GPU only. A CPU-only runtime raises an error.
- No `plan_fft`: cuPyNumeric plans inside its FFT task and exposes no plan.
- On lengths with a prime factor above 131 (e.g. 137), cuFFT is much slower
  and uses more memory. This is a cuFFT implementation detail. cuNumeric warns
  when it applies. Zero-padding to a power of two avoids it.

```@autodocs
Modules = [cuNumeric]
Pages = ["ndarray/fft.jl"]
```
