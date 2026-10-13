# Periodic Poisson (FFT)

## Solving the periodic Poisson equation with FFTs

We solve the Poisson equation on the unit square with periodic boundaries:

```math
\nabla^2 u(x,y) = f(x,y),
\qquad
f(x,y) = -8\pi^2\sin(2\pi x)\sin(2\pi y).
```

In Fourier space, the Laplacian becomes multiplication by
``\lambda_{k_x,k_y} = -4\pi^2(k_x^2+k_y^2)``. We recover the solution with a
forward FFT, division by these eigenvalues, and an inverse FFT:

```math
u = \mathcal{F}^{-1}\!\left[\frac{\mathcal{F}[f]}{\lambda}\right].
```

The zero-frequency eigenvalue is zero because the solution is determined
only up to an additive constant (``\nabla^2(u + C) = \nabla^2u``). We set that Fourier coefficient to zero
to choose the zero-mean solution (C = 0).

```julia
using cuNumeric

n = 64
ω = 2f0 * Float32(π)
xs = Float32.(0:(n - 1)) ./ Float32(n)

# Sample the prescribed forcing on the periodic grid.
f = NDArray([-2f0 * ω^2 * sin(ω * x) * sin(ω * y) for x in xs, y in xs])

# Laplacian eigenvalues.
k = NDArray(Float32.(vcat(0:(n ÷ 2 - 1), -(n ÷ 2):-1)))  # integer FFT frequencies
kx = cuNumeric.reshape(k, n, 1)
ky = cuNumeric.reshape(k, 1, n)
λ = (-ω^2) .* (kx.^2 .+ ky.^2)
λ[1:1, 1:1] .= 1f0 # insert placeholder, to avoid dividng by zero
λ = cuNumeric.as_type(λ, ComplexF32) # match `fft(f)` so the division doesn't promote

# Solve using cuNumeric.
u_hat = fft(f) ./ λ
u_hat[1:1, 1:1] .= 0f0  # Select the zero-mean solution (C = 0)
u = real(ifft(u_hat))

# Construct the analytical answer to check the result.
u_exact = [sin(ω * x) * sin(ω * y) for x in xs, y in xs]
error = fetch(maximum(abs.(u .- NDArray(u_exact))))
println("Maximum absolute error: ", error)
```

For this forcing, the exact zero-mean solution is

```math
u_{\mathrm{exact}}(x,y) = \sin(2\pi x)\sin(2\pi y).
```

These sine modes are resolved by the grid, so the FFT solve recovers the
analytical solution up to floating-point error.
