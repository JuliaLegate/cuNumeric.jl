# Diagonal and identity

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
X = D \ A                       # scale rows by 1 ./ d
Di = D + I                      # still Diagonal
C = A + I                       # dense NDArray
```

## Supported `Diagonal{<:NDArray}` APIs

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
- `det(D)` — `CNScalar` product of the diagonal

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
- `tr`, `sum`, `prod`, `maximum`, `minimum` — `CNScalar` (not a Julia scalar)
- `iszero`, `isone`, `istriu`, `istril`, `ishermitian`, `issymmetric`, `isposdef` — 0-dimensional `NDArray{Bool}`
- `opnorm(D)` / `opnorm(D, p)` for `p ∈ {1, 2, Inf}` — `CNScalar`
- `norm(D)` / `norm(D, p)` for finite `p` (including `±Inf`); off-diagonals are zero — `CNScalar`
- `cond(D)` / `cond(D, p)` for `p ∈ {1, 2, Inf}` — `CNScalar`
- `logdet(D)` for real `Diagonal` only — `CNScalar`

**Helpers on dense `NDArray`**

- `cuNumeric.diag` / `LinearAlgebra.diag` (2D → 1D), `cuNumeric.diagonal` (two
  axes of an `N`-D array → rank `N-1`), `cuNumeric.trace` / `LinearAlgebra.tr`
  (2D square → 0D)
- `iszero(A)` — all elements `== zero(T)` → 0-dimensional `NDArray{Bool}`
- `isone(A)` — square 2D vs `_eye(T, n)` → 0-dimensional `NDArray{Bool}` (non-square → `false`)

## Unsupported / fallthrough

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
