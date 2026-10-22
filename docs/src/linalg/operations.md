# Standard functions

## Matrix multiply

For two 2D arrays, `*` performs matrix multiplication. Use `.*` for an
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

## Vector operations

Supported `LinearAlgebra` operations include:

- `mul!(y, A, x)` and `mul!(y, A, x, α, β)` for matrix-vector products. `A * x` is also supported.
- Both `mul!` forms and `A * x` accept `Transpose(A)` and `Adjoint(A)` (or `A'`), including complex conjugation for the adjoint.
- `mul!(C, A, B, α, β)` for matrix-matrix products.
- `dot(x, y)` and entrywise `norm(A, p)` (dense-array norms currently require a GPU).
- `axpy!`, `axpby!`, `lmul!`, and `rmul!`.
- `ldiv!(y, D, x)` and `ldiv!(D, x)` for an NDArray-backed `Diagonal`.

`dot` and dense-array `norm` return [device scalars](../api_cnscalar.md).

## Transpose and adjoint wrappers

`Base.transpose(A)` (or `Transpose(A)`) and `adjoint(A)` (or `A'`) are lazy
wrappers around an NDArray vector or matrix. Matrix products accept either
wrapper on either side, including the three- and five-argument `mul!` forms.
Row-vector products and vector outer products are also supported:

```julia
C = A' * B
mul!(C, A', B, 2.0f0, 0.0f0)
x = cuNumeric.rand(Float32, size(A, 1))
y = A' * x
row = x' * A
outer = x * x'
```

Use `copy(A')` or `NDArray(A')` for an independent device array and `Array(A')`
or `collect(A')` for a host matrix. `collect(T, A')` converts the host result's
element type. Copying a wrapped vector preserves its row-vector wrapper;
`NDArray(x')` produces a 1 × n device matrix. `copyto!`, `transpose!`, and
`adjoint!` can write a transformed array into an existing NDArray.
Standard elementwise broadcasts, `fill!`, and broadcast assignment are
supported; writing into a wrapper updates its parent.

Unlike these lazy wrappers, `cuNumeric.transpose(A)` returns an NDArray view.

## Solve

`cuNumeric.solve(A, b)` solves a linear system and returns an array with the same
shape as `b`. `A` is a square `m × m` matrix and `b` has shape `(m,)` or
`(m, n)`. `A \ b` is equivalent.

```julia
A = cuNumeric.rand(Float32, 64, 64)
b = cuNumeric.rand(Float32, 64)
x = cuNumeric.solve(A, b)     # or: A \ b

B = cuNumeric.rand(Float32, 64, 4)
X = A \ B
```

## Helpers

These helpers live on `NDArray` and are also listed in the Public API:

- `cuNumeric.transpose`
- `cuNumeric.diag` (2D to 1D)
- `cuNumeric.trace`

## Iterative solvers

See the [Krylov.jl example](../examples/krylov.md) for conjugate gradient,
workspace reuse, and the scalar-handling macros.
