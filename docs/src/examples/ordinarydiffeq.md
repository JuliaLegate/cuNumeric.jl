# OrdinaryDiffEq.jl

Pass an `NDArray` initial state to a SciML `ODEProblem` and use array operations
in the right-hand side. The solver's state then stays in cuNumeric arrays.

> [!WARNING]
> Not all OrdinaryDiffEq solvers, callbacks, or solver options have been tested
> with `NDArray`.

## 2D heat diffusion

We solve the heat equation

```math
\frac{\partial u}{\partial t}
= \kappa \left(\frac{\partial^2 u}{\partial x^2}
+ \frac{\partial^2 u}{\partial y^2}\right)
```

on a square grid with zero temperature at the boundaries. The initial
temperature is a smooth sine profile, and a five-point finite-difference
Laplacian approximates diffusion in the interior.

The grid spacing is `DX = 1`, so an `n-by-n` grid covers a square of
side length `(n - 1) * DX`. Increasing `n` expands the domain at the same resolution.

In an environment with cuNumeric installed, add the solver packages:

```julia
using Pkg
Pkg.add(["SciMLBase", "OrdinaryDiffEqLowStorageRK"])
```

This complete example advances a `128-by-128` grid from `t = 0` to `t = 1`
with diffusivity `0.2` and a fixed time step of `0.05`:

```julia
using cuNumeric
using OrdinaryDiffEqLowStorageRK: CarpenterKennedy2N54
using SciMLBase: ODEProblem, solve, FullSpecialize

const DX = 1

heat_slice(u, rows, cols) = @views u[rows, cols]

function heat!(du, u, κ, t)
    n = size(u, 1)
    fill!(du, zero(eltype(du)))
    interior = (κ / DX^2) .* (
        heat_slice(u, 3:n, 2:(n - 1)) .+
        heat_slice(u, 1:(n - 2), 2:(n - 1)) .+
        heat_slice(u, 2:(n - 1), 3:n) .+
        heat_slice(u, 2:(n - 1), 1:(n - 2)) .-
        4 .* heat_slice(u, 2:(n - 1), 2:(n - 1))
    )
    copyto!(view(du, 2:(n - 1), 2:(n - 1)), interior)
    return nothing
end

n = 128
s = Float32.(sin.(range(0, pi; length=n)))
s[1] = s[end] = 0f0
u0 = NDArray(s * transpose(s))
problem = ODEProblem{true,FullSpecialize}(heat!, u0, (0f0, 1f0), 0.2f0)
solution = solve(problem, CarpenterKennedy2N54(williamson_condition=false);
                 dt=0.05f0, save_everystep=false,
                 unstable_check=(dt, u, p, t) -> false)

@show typeof(solution.u[end])
```

The right-hand side fills `du` with zeros, then updates only its interior.
Since the initial boundary values are zero and their derivatives stay zero,
the boundary temperature remains fixed. The final state,
`solution.u[end]`, is an `NDArray{Float32,2}`.

## Solver options

This example uses
[`CarpenterKennedy2N54`](https://docs.sciml.ai/OrdinaryDiffEq/stable/explicit/LowStorageRK/#OrdinaryDiffEqLowStorageRK.CarpenterKennedy2N54),
a fourth-order low-storage Runge-Kutta method with fixed time steps.

- `FullSpecialize` lets the problem work with both the initial `NDArray` and
  the solver's work arrays.
- `williamson_condition=false` disables an optimization that only supports
  ordinary Julia `Array` storage.
- `save_everystep=false` avoids retaining every intermediate time step.
- `unstable_check=(dt, u, p, t) -> false` disables the default per-step
  instability scan, which can loop over individual elements of custom arrays.
  This also removes that automatic check: validate the result separately and
  choose a stable time step when changing the grid spacing or diffusivity.

The [benchmark version](https://github.com/JuliaLegate/benchmarking/tree/main/composability/ordinarydiffeq)
checks the final state against the known decay of the discrete sine mode.
Its run notes report this example on an H100 with Julia 1.13.
