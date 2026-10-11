# Krylov.jl

Use Krylov's conjugate gradient solver with an `NDArray` matrix and vectors.
Loading `Krylov` alongside cuNumeric activates the package extension.

## Conjugate gradient

For a symmetric positive-definite system, solve `A * x = b` with:

```julia
using cuNumeric, Krylov

A = NDArray([4.0 1.0; 1.0 3.0])
b = NDArray([1.0, 2.0])
x, stats = @allowfetch Krylov.cg(A, b; rtol=1e-8)
```

`@allowfetch` lets the solver retrieve the scalar values it needs for convergence
decisions. The solution `x` remains an `NDArray`.

## Reuse a workspace

Allocate a workspace once, then use `cg!` for repeated solves with compatible
matrix and vector sizes and element types:

```julia
workspace = Krylov.CgWorkspace(A, b)
@allowfetch Krylov.cg!(workspace, A, b; rtol=1e-8)

x = workspace.x
stats = workspace.stats

b2 = NDArray([2.0, 1.0])
@allowfetch Krylov.cg!(workspace, A, b2; rtol=1e-8)
```

Each call updates `workspace.x` and `workspace.stats`. Copy the solution before
the next solve if you need to retain it.

## Complex systems

For Hermitian positive-definite complex systems, also use `@allowpromotion`.
CG computes real scalar coefficients that need to promote when scaling complex
vectors. Wrap the solver call in both macros:

```julia
@allowpromotion @allowfetch Krylov.cg!(workspace, A, b; rtol=1e-8)
```

The same macros apply to the allocating `Krylov.cg(A, b)` call.

The benchmarking repository contains a larger
[Krylov example and run instructions](https://github.com/JuliaLegate/benchmarking/tree/main/composability/krylov)
comparing the stock solvers on cuNumeric, CUDA.jl, and Dagger arrays.
