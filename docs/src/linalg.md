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

There is no public dense-matrix `cholesky`, `eig`, `lu`, matrix `inv`, or
`ldiv!` yet (beyond the `Diagonal` / `NDArray` paths listed above).
Elementwise `inv` / `^-1` are unary operations, not matrix inverse.
