# FFT

cuNumeric.jl exposes GPU-only [`fft`](@ref) / [`ifft`](@ref) (and in-place
[`fft!`](@ref) / [`ifft!`](@ref)) on `NDArray`. The functions follow
[AbstractFFTs.jl](https://github.com/JuliaMath/AbstractFFTs.jl) / FFTW
conventions, not NumPy's defaults:

- every dimension is transformed unless you pass `dims`
- `fft` is unnormalized; `ifft` divides by the product of the transformed lengths
- real input is promoted to complex (`Float32` → `ComplexF32`, otherwise `ComplexF64`)

There is no `plan_fft`, and we cannot control the cuFFT plan. cupynumeric never
returns a `cufftHandle` (or any other plan object). Each `fft` / `ifft` call
launches the `CUPYNUMERIC_FFT` task; cuFFT planning and any internal plan cache
live entirely inside that GPU task. A Julia `Plan` that only stored sizes and
re-called `fft` would not be a real plan, so `plan_fft` is not implemented.
`using AbstractFFTs; fft(A)` dispatches, but `plan_fft` will not.

```julia
using AbstractFFTs
using cuNumeric

A = cuNumeric.rand(ComplexF32, 64, 64)
Y = fft(A)            # all dimensions
Z = fft(A, 1)         # first dimension only
ifft(Y)               # ≈ A

fft!(copy(A))         # overwrites a complex array
```

`fft!` / `ifft!` require `ComplexF32` or `ComplexF64`. Real arrays must go
through out-of-place `fft`.

A stack of independent transforms uses [`batched_fft`](@ref): the leading
dimension is the batch, and every trailing dimension is transformed. That is
the same task as `fft(A, 2:ndims(A))`. The batch axis is the one that can
split across GPUs; an all-axes `fft(A)` cannot.

```julia
signals = cuNumeric.rand(ComplexF32, 32, 1024)   # 32 length-1024 traces
batched_fft(signals)                             # FFT along dim 2

fields = cuNumeric.rand(ComplexF32, 8, 64, 64)   # 8 images
batched_fft(fields)                              # 2-d FFT of each
```

FFT is GPU-only. A CPU-only runtime raises an error. Multi-GPU use is limited
to batching over dimensions that are not transformed (the same restriction as
cupynumeric).

Awkward lengths whose prime factors exceed 131 make cuFFT take the Bluestein
path; a warning is emitted. Padding to a nearby highly composite size avoids
that.

```@docs
fft
ifft
fft!
ifft!
batched_fft
batched_ifft
batched_fft!
batched_ifft!
```
