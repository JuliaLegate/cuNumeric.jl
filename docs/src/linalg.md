# Linear Algebra

cuNumeric.jl provides matrix multiplication, solves, Cholesky, eigen, SVD, QR,
and related helpers for `NDArray`.

All of the decompositions accept `Float32`, `Float64`, `ComplexF32`, and
`ComplexF64`. Integer and `Bool` inputs are converted to `Float64`. As
everywhere else in the package, that conversion needs `@allowpromotion` (or
`allowpromotion`) only when it widens the element type, so `Int64` and `UInt64`
pass through silently while `Int32`, smaller integers, and `Bool` do not.

## Which entry point to use

Single matrices go through the standard `LinearAlgebra` functions and return the
standard factorization objects. Stacks of matrices have no `LinearAlgebra`
equivalent, so they get their own `batched_*` names.

| Single matrix (2D) | Returns | Stack of matrices (3D) |
| --- | --- | --- |
| `LinearAlgebra.cholesky(A)` | `Cholesky` | `cuNumeric.batched_cholesky(A)` |
| `LinearAlgebra.eigen(A)` | `Eigen` | `cuNumeric.batched_eigen(A)` |
| `LinearAlgebra.eigvals(A)` | `NDArray` | `cuNumeric.batched_eigvals(A)` |
| `LinearAlgebra.svd(A)` | `SVD` | not supported |
| `LinearAlgebra.qr(A)` | `NDArrayQR` | not supported |
| `cuNumeric.solve(A, b)`, `A \ b` | `NDArray` | `cuNumeric.batched_solve(A, B)` |

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

## Cholesky

`LinearAlgebra.cholesky(A)` returns a `Cholesky` object holding the lower factor
`L`, with `A ≈ L * L'`.

```julia
using LinearAlgebra

A = cuNumeric.rand(Float32, 64, 64)
A = A * cuNumeric.transpose(A) + 64 * NDArray{Float32}(I, 64, 64)  # make it SPD

F = cholesky(A)
L = F.L          # LowerTriangular view of F.factors
```

Three things differ from Base:

- The input is **not** checked for being Hermitian; only its lower triangle is
  read. There are no `Hermitian` / `Symmetric` overloads.
- A non-positive-definite input raises an `ErrorException` carrying the task's
  `"Matrix is not positive definite"` message, not a `PosDefException`. The
  `check` keyword is therefore not supported.
- `F.U` (and hence destructuring as `L, U = F`) needs `copy(F.factors')`, which
  falls back to scalar indexing until `adjoint(::NDArray)` is implemented. Use
  `F.L` or `F.factors`.

## Eigendecomposition

`LinearAlgebra.eigen(A)` returns an `Eigen` object; `eigvals` and `eigvecs`
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

The symmetric/Hermitian path (`eigh`, backed by the `SYEV` task) is not wired up
yet; it needs `Hermitian` / `Symmetric` support on `NDArray` first.

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

Note that `full=false` is the Julia default, whereas numpy's
`full_matrices=True` is not. Destructuring as `U, S, V = F` also gives you the
*adjoint* of `F.Vt`, and `F.V` is a lazy `Adjoint` wrapper, so operating on it
falls back to scalar indexing until `adjoint(::NDArray)` is implemented. Prefer
`F.Vt`.

`S` is real-valued for both real and complex inputs.

## QR decomposition

`LinearAlgebra.qr(A)` returns an `NDArrayQR`, holding the economy-size factors
with `A ≈ F.Q * F.R`. For a 2D `m × n` input and `k = min(m, n)`, `Q` is `m × k`
and `R` is `k × n`.

```julia
A = cuNumeric.rand(Float32, 128, 64)
F = qr(A)
Q, R = F                 # or: F.Q, F.R
```

`NDArrayQR` is a `LinearAlgebra.Factorization` but not one of Base's QR types.
Base's default `QRCompactWY` stores a blocked Householder representation and
`QR` stores `factors` plus `τ`; the backend runs `geqrf` followed by `orgqr` and
discards `τ`, so neither is constructible. The practical difference is that
`F.Q` is a materialized `NDArray` rather than a lazy `QRCompactWYQ`.

SVD and QR accept only 2D arrays; there are no batched versions.

## Batched decompositions

The `batched_*` functions apply an operation to every matrix in a stack. They
take exactly one batch dimension, i.e. shape `(b, m, m)`, and return plain
tuples or arrays rather than factorization objects.

```julia
As = cuNumeric.rand(Float32, 8, 32, 32)
Bs = cuNumeric.rand(Float32, 8, 32, 2)

Xs = cuNumeric.batched_solve(As, Bs)         # (8, 32, 2)
Ls = cuNumeric.batched_cholesky(As)          # (8, 32, 32), needs SPD blocks
values, vectors = cuNumeric.batched_eigen(As)  # (8, 32) and (8, 32, 32)
```

Each matrix is factored on a single processor, so this is a good fit for many
small matrices and a poor one for a few large ones. Two or more batch dimensions
are rejected: the `POTRF` task body is only instantiated up to three dimensions,
and Legate.jl builds launch domains for at most three dimensions.

Every caveat listed under Cholesky and Eigendecomposition applies here too.

## Helpers

These helpers live on `NDArray` and are also listed in the Public API:

- `cuNumeric.transpose`
- `cuNumeric.diag` (2D to 1D)
- `cuNumeric.trace`

## Diagonal and identity

Prefer structured `LinearAlgebra` types over materializing a full matrix.

Wrap a 1D `NDArray` in `Diagonal` for scale / solve / inverse along a diagonal.
Use `LinearAlgebra.I` (`UniformScaling`) for `A + I`, `D + I`, and `A * I`.
`D + I` stays a `Diagonal`; `A + I` returns a dense `NDArray`.

```julia
using LinearAlgebra
using cuNumeric

d = cuNumeric.ones(Float32, 64)
D = Diagonal(d)                 # preferred: keep diagonal structure
A = cuNumeric.rand(Float32, 64, 64)
v = cuNumeric.rand(Float32, 64)

y = D * v                       # scale a vector
B = D * A                       # scale rows
X = D \ A                       # scale columns by 1 ./ d
Di = D + I                      # still Diagonal
C = A + I                       # dense NDArray
```

### Supported `Diagonal{<:NDArray}` APIs

These paths stay on-device (no host densify for the math). RHS / other operands
must be `NDArray` unless noted.

**Construction / display**

- `Diagonal(v::NDArray{<:Any,1})` — wrap without copying
- `Diagonal(A::NDArray{<:Any,2})` — `Diagonal(diag(A))`
- `Matrix(D)` / `Matrix{T}(D)` — densify to a host `Matrix` (conversion only)
- `show` — densifies `.diag` for printing only

**Multiply / divide / inverse**

- `D * A`, `A * D`, `D * v` for 2D / 1D `NDArray`
- `mul!`, `lmul!`, `rmul!` with `NDArray`
- `D \ B`, `A / D`, `ldiv!`, `rdiv!` with `NDArray`
- `inv(D)` — reciprocal on-device; zeros become Inf (no `SingularException`)
- `det(D)` — 0-dimensional `NDArray` product of the diagonal

**`NDArray` ± `Diagonal`**

- `A + D`, `D + A`, `A - D`, `D - A` for square 2D `NDArray`

**UniformScaling (`I`)**

- `NDArray{T}(I, m, n)` / `NDArray(I, …)`, `copyto!(A, I)`, `one(A)`, `oneunit(A)`
- `A ± I`, `A * I`, `I * A`
- `D ± I`, `D * I`, `I * D`, `copyto!(D, I)` — `D ± I` stays `Diagonal`

**Broadcast**

- Structure- or zero-preserving broadcasts on `Diagonal` (e.g. `D .* c`,
  `D .*= c`, `D .+ D`) lower to 1D broadcast on `.diag`
- Densifying out-of-place broadcasts (e.g. `D .+ 1`, `D .+ A`) materialize a
  dense `NDArray`, matching Base’s densify-to-`Matrix` behavior
- In-place densifying writes into `Diagonal` (e.g. `D .+= 1`, `D .+= A`) still
  throw `ArgumentError` (off-diagonal / densify), matching Base

**Eigen / reductions / predicates / norms**

- `eigvals(D)`, `eigen(D)`, `eigvecs(D)` — unsorted; values are a copy of the
  diagonal (`NDArray`), vectors are `NDArray` identity. Keyword `sortby` is not
  supported on this method.
- `tr`, `sum`, `prod`, `maximum`, `minimum` — 0-dimensional `NDArray` (not a Julia scalar)
- `iszero`, `isone`, `istriu`, `istril`, `ishermitian`, `issymmetric`, `isposdef` — 0-dimensional `NDArray{Bool}`
- `opnorm(D)` / `opnorm(D, p)` for `p ∈ {1, 2, Inf}` — 0-dimensional `NDArray`
- `norm(D)` / `norm(D, p)` for finite `p` (including `±Inf`); off-diagonals are zero — 0-dimensional `NDArray`
- `cond(D)` / `cond(D, p)` for `p ∈ {1, 2, Inf}` — 0-dimensional `NDArray`
- `logdet(D)` for real `Diagonal` only — 0-dimensional `NDArray`

**Helpers on dense `NDArray`**

- `cuNumeric.diag` / `LinearAlgebra.diag` (2D → 1D), `cuNumeric.trace` / `LinearAlgebra.tr` (2D square → 0D)
- `iszero(A)` — all elements `== zero(T)` → 0-dimensional `NDArray{Bool}`
- `isone(A)` — square 2D vs `_eye(T, n)` → 0-dimensional `NDArray{Bool}` (non-square → `false`)

### Unsupported / fallthrough

Other `LinearAlgebra` operations on `Diagonal{<:NDArray}` (for example `svd`,
`svdvals`, `pinv`, `logabsdet`, complex `logdet`, `kron`, `cholesky`, host
`AbstractArray` RHS for `\` / `/` / `ldiv!` / `rdiv!`, or `eigen(...; sortby)`)
are **not** specially implemented. They fall through to Base and typically fail
with the package’s scalar-indexing error (NDArray does not support scalar
indexing without `@allowscalar`). There are no “not implemented” `ArgumentError`
stubs for these.

Only densify when you truly need a full identity matrix:

```julia
E = NDArray{Float32}(I, 64, 64) # dense identity
copyto!(A, I)                   # fill an existing array
E2 = one(A)                     # same shape / eltype as A
```

Avoid building a dense identity (or densifying `D` with `Matrix(D)`) just to
scale or shift; prefer `Diagonal` and `I` instead. There is no public `eye`.

## Not available yet

There is no public dense-matrix `lu`, matrix `inv`, or `ldiv!` yet (beyond the
`Diagonal` / `NDArray` paths listed above). Elementwise `inv` / `^-1` are unary
operations, not matrix inverse.

Also missing: `eigh` / Hermitian eigen (needs `Hermitian` and `Symmetric`
support on `NDArray`), batched SVD and QR, and the multi-GPU cuSolverMp paths
for Cholesky and solve.
