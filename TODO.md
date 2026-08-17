## TO-DO List of Missing Important Features
- Implement `unary_reduction` over arbitrary dims
- Out-parameter `binary_op`
- Replace `as_type` with `Base.convert`
- Support Ints on methods that takes floats
- Programatic manipulation of Legate hardware config (not currently possible)
- Add Aqua.jl to CI to ensure we didn't pirate any types

## Base

Easy `Base` / `AbstractArray` gaps for `NDArray`. Module helpers
(`cuNumeric.reshape` / `transpose` / `unique` / …) often exist, but the
corresponding `Base` methods are missing, so calls fall through to
`AbstractArray` and may scalar-index. Wire up `Base.*` when convenient.

**P0**
- `Base.reshape`, `Base.vec`
- `fill!` (convertible eltypes)
- `collect`

**P0 done**
- `iszero` / `isone` (on-device reductions → `Bool` via scalar sync; `isone` square 2D only)

**P1**
- `transpose` / `adjoint`
- `unique`
- `ones_like`
- `dropdims`

**P2**
- `extrema`, `mean`
- `count` / nonzero
- `abs2`
- numeric `all` / `any`
- `sum(f, A)` specials

**P3**
- `copyto!(NDArray, AbstractArray)`
- `floor` / `ceil` / `clamp`
- 2D `permutedims`
- `diff`

## LinearAlgebra

Starter list of easy/medium LA gaps. Prefer wiring `LinearAlgebra` entry
points so they do not fall through to scalar-indexing Base paths.

**BLAS-1 style (dense `NDArray`)**
- `axpy!` / `axpby!` — common scale-and-add; examples use broadcast today
- `scal!` and related in-place scale
- `LinearAlgebra.dot` if not already Base-wired to `nda_dot`

**Reductions / traces**
- `LinearAlgebra.tr` for dense 2D `NDArray` — `cuNumeric.trace` exists; `tr` is
  already wired for `Diagonal{<:NDArray}`

**Diagonal vs fallthrough (context)**
- Already on-device for `Diagonal`: `mul!` / `lmul!` / `rmul!`, `\` / `/`,
  `tr`, `norm` / `opnorm`, many predicates — see `docs/src/linalg.md`
- Still fallthrough / unsupported on `Diagonal` (e.g. `svd`, `pinv`,
  `cholesky`, host `AbstractArray` RHS): leave alone unless fixing is cheap;
  densify intentionally when needed

**Decompositions**
- Done: `cholesky`, `eigen` / `eigvals` / `eigvecs`, `svd`, `qr`, and `\` are
  wired to `LinearAlgebra` for 2D `NDArray`; `batched_solve` / `batched_cholesky`
  / `batched_eigen` / `batched_eigvals` cover one batch dimension
- `eigh` / Hermitian eigen — the `SYEV` task is already wrapped, but there is no
  entry point until `Hermitian` / `Symmetric` work on `NDArray`
- `PosDefException` from `cholesky` — a non-positive-definite input now raises a
  catchable `ErrorException` (since the launchers call `task_throws_exception`),
  but mapping it onto `LinearAlgebra.PosDefException` needs the failing pivot
  index, which the task message does not carry
- The decomposition launchers must keep calling `task_throws_exception`, as
  cupynumeric's own launchers do. Besides propagating task exceptions it grows
  each leaf allocation pool by `--max-exception-size` (4096 bytes), which the
  batched GPU kernels depend on: `CuPyNumericMapper::allocation_pool_size` only
  declares one 16-byte-aligned `int32` of ZCMEM for `SOLVE` / `GEEV`, while
  `solve.cu` allocates `batchsize` pointer arrays plus `batchsize` infos and
  `geev.cu` allocates an info per matrix. Without it, `b > 1` aborts on GPU
- `adjoint(::NDArray)` (already on the P1 list) would unblock `Cholesky.U`,
  `SVD.V`, destructuring a `Cholesky` as `L, U`, and `ldiv!` / least-squares `\`
  on `NDArrayQR` (which needs `Q' * b`)
- More than one batch dimension — needs `domain_from_shape` in Legate.jl to
  handle more than three dimensions, plus a cupynumeric `POTRF` instantiated for
  `DIM >= 4`
- Multi-GPU cuSolverMp paths (`MP_POTRF`, `MP_SOLVE`) for large single matrices;
  `cupynumeric_has_cusolvermp()` gates them and they need an NCCL communicator
- Batched `svd` / `qr`
