# Periodic Poisson (FFT)

The Poisson equation

```math
\nabla^2 u = f
```

on the periodic unit square is a pointwise divide in Fourier space. With
``k = 2\pi (m_x, m_y)`` the wavevector of each DFT mode,

```math
\hat{u}[k] = \frac{\hat{f}[k]}{-|k|^2}, \qquad \hat{u}[0] = 0.
```

The zero mode is dropped so ``u`` has mean zero (the potential is only defined
up to a constant). One forward [`fft`](@ref), a broadcasted divide, and one
[`ifft`](@ref) is the whole solve. That is the electrostatics / gravitational
potential of a periodic charge density, not an FFT round-trip. There is no
reusable plan to hold across the two transforms; each call launches
`CUPYNUMERIC_FFT` and cuFFT planning stays inside that task.

A manufactured solution ``u = \sin(2\pi x)\sin(2\pi y)`` has
``\nabla^2 u = -8\pi^2 u``, which is what the example checks.

```julia
# found in examples/poisson_fft.jl
using cuNumeric

function integer_fftfreq(n::Int)
    n2 = n ÷ 2
    return iseven(n) ? vcat(0:(n2 - 1), (-n2):-1) : vcat(0:n2, (-n2):-1)
end

function poisson_inv_laplacian(::Type{T}, n::Int) where {T}
    freq = integer_fftfreq(n)
    invk = Matrix{T}(undef, n, n)
    s = T(4 * π^2)
    for j in 1:n, i in 1:n
        k2 = s * T(freq[i]^2 + freq[j]^2)
        invk[i, j] = k2 == 0 ? zero(T) : -inv(k2)
    end
    return invk
end

n = 64
xs = range(0, 1; length=n + 1)[1:(end - 1)]
u_true = Float32[sin(2π * x) * sin(2π * y) for x in xs, y in xs]
f = NDArray((-8 * Float32(π)^2) .* u_true)

invk = NDArray(poisson_inv_laplacian(Float32, n))
u = real(ifft(fft(f) .* invk))
```

A stack of right-hand sides (several charge distributions, several snapshots)
uses [`batched_fft`](@ref). The leading axis is the batch and is the one that
can split across GPUs; a single all-axes `fft` of one grid cannot.

```julia
f_batch = NDArray(repeat(reshape(Array(f), 1, n, n), 4, 1, 1))
invk3 = reshape(invk, 1, n, n)
u_batch = real(cuNumeric.batched_ifft(cuNumeric.batched_fft(f_batch) .* invk3))
```
