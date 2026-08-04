# Linear Algebra

cuNumeric.jl provides matrix multiplication, batched solves, SVD, QR, and related
helpers for `NDArray`.

`solve`, `svd`, and `qr` accept `Float32`, `Float64`, `ComplexF32`, and
`ComplexF64`. Integer and `Bool` inputs require `@allowpromotion` or
`allowpromotion` and produce `Float64` outputs.

## Matrix multiply

For two 2D arrays, `*` performs matrix multiplication; use `.*` for an
elementwise product.

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

`cuNumeric.solve(A, b)` solves linear systems and returns an array with the same
shape as `b`. `A` has shape `(..., m, m)` and `b` has shape `(..., m)` or
`(..., m, n)`.

```julia
A = cuNumeric.rand(Float32, 64, 64)
b = cuNumeric.rand(Float32, 64)
x = cuNumeric.solve(A, b)

B = cuNumeric.rand(Float32, 64, 4)
X = cuNumeric.solve(A, B)

As = cuNumeric.rand(Float32, 8, 32, 32)
Bs = cuNumeric.rand(Float32, 8, 32, 2)
Xs = cuNumeric.solve(As, Bs)
```

## Singular value decomposition

`cuNumeric.svd(A, full_matrices=true)` returns `(U, S, Vh)` for a 2D `m × n`
array. With `k = min(m, n)`, the output shapes are:

- Full: `U` is `m × m`, `S` has length `k`, and `Vh` is `n × n`.
- Thin: `U` is `m × k`, `S` has length `k`, and `Vh` is `k × n`.

```julia
A = cuNumeric.rand(Float32, 128, 64)
U, S, Vh = cuNumeric.svd(A, false)
```

`S` is real-valued for both real and complex inputs.

## QR decomposition

`cuNumeric.qr(A)` returns the economy-size factors `(Q, R)` for a 2D `m × n`
array. With `k = min(m, n)`, `Q` is `m × k` and `R` is `k × n`.

```julia
A = cuNumeric.rand(Float32, 128, 64)
Q, R = cuNumeric.qr(A)
```

SVD and QR currently accept only 2D arrays; batched decompositions are not
supported.

## Helpers

These helpers live on `NDArray` and are also listed in the Public API:

- `cuNumeric.transpose`
- `cuNumeric.eye`
- `cuNumeric.diag` (2D to 1D)
- `cuNumeric.trace`

## Not available yet

There is no public `cholesky`, `eig`, `lu`, matrix `inv`, or `ldiv!` yet.
Elementwise `inv` / `^-1` are unary operations, not matrix inverse.
