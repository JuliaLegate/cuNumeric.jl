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

Matrix operands may also be lazy `Transpose` or `Adjoint` wrappers around
NDArrays. Both `*` and the three- and five-argument `mul!` forms support a
wrapped matrix on either side, including two wrapped matrices. A wrapped
matrix can also multiply an NDArray vector.

```julia
AtB = transpose(A) * B
AhB = A' * B
mul!(similar(AhB), A', B, 2f0, 0f0)
```

These operations use the cuPyNumeric contraction backend. Transpose is
represented by permuted contraction labels and backend store views; complex
adjoints additionally allocate a temporary for bulk conjugation. Real adjoints
need no conjugation temporary. The destination of `mul!` must not share storage
with either input, including the parent of a wrapper.

`transpose(A)` is Julia's lazy wrapper; `cuNumeric.transpose(A)` is the existing
backend helper returning an NDArray. This support covers matrix wrappers in
multiplication, not general wrapper broadcasting, slicing, or row-vector products.

```@autodocs
Modules = [cuNumeric]
Pages = ["ndarray/binary.jl"]
Filter = t -> t isa Function && nameof(t) === :mul!
```

## Vector operations

Supported `LinearAlgebra` operations include:

- `mul!(y, A, x)` and `mul!(y, A, x, α, β)` for matrix-vector products. `A * x` is also supported.
- `mul!(C, A, B, α, β)` for matrix-matrix products.
- `dot(x, y)` and entrywise `norm(A, p)` (dense-array norms currently require a GPU).
- `axpy!`, `axpby!`, `lmul!`, and `rmul!`.
- `ldiv!(y, D, x)` and `ldiv!(D, x)` for an NDArray-backed `Diagonal`.

`dot` and dense-array `norm` return [device scalars](../api_cnscalar.md).

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
