# Factorizations

These methods take a single 2D `NDArray`. See the
[overview](../linalg.md) for supported element types and
[Batched operations](batched.md) for stacks of matrices.

```julia
using cuNumeric, LinearAlgebra
```

## Cholesky

`LinearAlgebra.cholesky(A)` returns a `Cholesky` object holding the lower factor
`L`, with `A ≈ L * L'`.

```julia
A = cuNumeric.rand(Float32, 64, 64)
A = A * cuNumeric.transpose(A) + 64 * NDArray{Float32}(I, 64, 64)  # make it SPD

F = cholesky(A)
L = F.L          # LowerTriangular view of F.factors
```

Three things differ from Base:

- The input is **not** checked for being Hermitian. Only its lower triangle is
  read. There are no `Hermitian` / `Symmetric` overloads.
- A non-positive-definite input raises an `ErrorException` carrying the task's
  `"Matrix is not positive definite"` message, not a `PosDefException`. The
  `check` keyword is therefore not supported.
- `F.U` (and hence destructuring as `L, U = F`) needs `copy(F.factors')`, which
  falls back to scalar indexing until `adjoint(::NDArray)` is implemented. Use
  `F.L` or `F.factors`.

## Eigendecomposition

`LinearAlgebra.eigen(A)` returns an `Eigen` object. `eigvals` and `eigvecs`
return the pieces individually. Column `j` of the eigenvector matrix is the
eigenvector for eigenvalue `j`.

```julia
A = cuNumeric.rand(Float64, 64, 64)

F = eigen(A)
values, vectors = F      # or: eigvals(A), eigvecs(A)
```

Eigenvalues and eigenvectors are **always complex**, even when `A` is real with
real eigenvalues: `ComplexF32` for `Float32` / `ComplexF32` input and
`ComplexF64` otherwise. This follows the underlying LAPACK `geev` path and
differs from Base, which returns real factors for such input. CUDA.jl's
non-symmetric branch behaves the same way.

On a GPU this requires `cusolverDnXgeev`, added in CUDA 12.6.2. When it is
missing, the eigen entry points throw an error rather than silently falling back
to host execution.

Symmetric and Hermitian eigendecomposition (`eigh`) is not supported yet.
It requires `Hermitian` / `Symmetric` support on `NDArray`.

## Singular value decomposition

`LinearAlgebra.svd(A; full=false)` returns an `SVD` object with
`A ≈ F.U * Diagonal(F.S) * F.Vt`.

```julia
A = cuNumeric.rand(Float32, 128, 64)
F = svd(A)
U, S, Vt = F.U, F.S, F.Vt
```

With `k = min(m, n)`, the output shapes are:

- Thin (`full=false`, the default): `U` is `m × k`, `S` has length `k`, and `Vt`
  is `k × n`.
- Full (`full=true`): `U` is `m × m`, `S` has length `k`, and `Vt` is `n × n`.

The default `full=false` returns thin factors. Destructuring as `U, S, V = F` also gives you the
*adjoint* of `F.Vt`, and `F.V` is a lazy `Adjoint` wrapper. Matrix-vector
`mul!(y, F.V, x)` supports this wrapper; copying it and matrix-matrix products
still fall back to scalar indexing. Prefer `F.Vt` for those operations.

`S` is real-valued for both real and complex inputs.

The backend only factors tall or square matrices (`m >= n`), matching
cuPyNumeric. A wide input throws `ArgumentError`.

## QR decomposition

`LinearAlgebra.qr(A)` returns an `NDArrayQR`, holding the economy-size factors
with `A ≈ F.Q * F.R`. For a 2D `m × n` input and `k = min(m, n)`, `Q` is `m × k`
and `R` is `k × n`.

```julia
A = cuNumeric.rand(Float32, 128, 64)
F = qr(A)
Q, R = F                 # or: F.Q, F.R
```

`NDArrayQR` is a `LinearAlgebra.Factorization` with explicit `Q` and `R` arrays.
Unlike Base's default QR factorization, `F.Q` is a materialized `NDArray`
rather than a lazy `QRCompactWYQ`.

SVD and QR accept only 2D arrays. There are no batched versions.
