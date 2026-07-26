# Linear Algebra

cuNumeric.jl supports a small set of linear algebra operations on `NDArray`. This page covers matrix multiply, batched solve, and related helpers. Related autodocs also appear under [NDArray Reference](./api.md).

## Matrix multiply

For two 2D arrays, `*` is matrix multiplication (GEMM), not elementwise multiply. Use `.*` when you want an elementwise product of matrices.

```julia
using LinearAlgebra
using cuNumeric

A = cuNumeric.rand(Float32, 128, 128)
B = cuNumeric.rand(Float32, 128, 128)

C = A * B                 # allocates a new result
mul!(similar(C), A, B)    # in-place GEMM into an existing array
```

```@autodocs
Modules = [cuNumeric]
Pages = ["ndarray/binary.jl"]
Filter = t -> t isa Function && nameof(t) === :mul!
```

## Solve (batched)

`cuNumeric.solve(A, b)` solves linear systems. It is not Julia's `\`.

```@docs
cuNumeric.solve
```

Shapes follow the batched signature:

- `A` is `(..., m, m)` (last two dims square)
- `b` is `(..., m)` or `(..., m, n)`
- result is `(..., m)` or `(..., m, n)`

A 1D right-hand side is reshaped internally to a single column, then reshaped back.

```julia
using cuNumeric

# Single system: (m, m) and (m,)
A = cuNumeric.rand(Float32, 64, 64)
b = cuNumeric.rand(Float32, 64)
x = cuNumeric.solve(A, b)

# Several right-hand sides: (m, m) and (m, n)
B = cuNumeric.rand(Float32, 64, 4)
X = cuNumeric.solve(A, B)

# Batched systems: (batch, m, m) and (batch, m, n)
As = cuNumeric.rand(Float32, 8, 32, 32)
Bs = cuNumeric.rand(Float32, 8, 32, 2)
Xs = cuNumeric.solve(As, Bs)
```

Notes:

- Accepted types: `Float32`, `Float64`, `ComplexF32`, `ComplexF64`. Integer or `Bool` inputs promote to `Float64` only when promotion is allowed (`@allowpromotion` / `allowpromotion`).
- The implementation always goes through a batched Legate `SOLVE` task, including the 2D case.
- Batch dimensions are supported in the API. Coverage for higher-rank batches in the test suite is still thin, so start with 2D and small batches when validating new code.

## Helpers

These helpers live on `NDArray` and are also listed in the Public API:

- `cuNumeric.transpose`
- `cuNumeric.eye`
- `cuNumeric.diag` (2D to 1D)
- `cuNumeric.trace`

## Not available yet

There is no public `svd`, `qr`, `cholesky`, `eig`, `lu`, matrix `inv`, or `ldiv!` in cuNumeric.jl yet. Elementwise `inv` / `^-1` exist as unary ops; those are not matrix inverse.
