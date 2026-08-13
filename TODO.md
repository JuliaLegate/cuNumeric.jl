## TO-DO List of Missing Important Features
- Implement `unary_reduction` over arbitrary dims
- Out-parameter `binary_op`
- Replace `as_type` with `Base.convert`
- Support Ints on methods that takes floats
- Programatic manipulation of Legate hardware config (not currently possible)
- Float32 random number generation (not possible in current C++ API)
- Normal random numbers (not possible in current C++ API)
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
