# Map and Mapped Reductions

## `map` and `map!`

`map(f, A)` and `map(f, A, B)` are the same as `f.(A)` and `f.(A, B)`.
`map!(op, dest, A, B)` writes into `dest` for built-in binary operators
(`+`, `*`, `max`, …). All three arrays must have the same shape.

```julia
A, B = cuNumeric.rand(1024), cuNumeric.rand(1024)
C = map(sin, A)
map!(*, C, A, B)
```

```@docs
Base.map(::Function, ::NDArray)
Base.map(::Function, ::NDArray{A,N}, ::NDArray{B,N}) where {A,B,N}
Base.map!(::Function, ::NDArray, ::NDArray, ::NDArray)
```

## `mapreduce`

`mapreduce(f, op, A; dims=:, init=...)` maps and reduces an NDArray in one
pass, without a mapped copy. `op` can be `+`, `*`, `min` or `max`, or `&`/`|`
when the mapping returns `Bool`. Mapped `sum`,
`prod`, `minimum`, `maximum`, `all`, `any`, `count`, and `mean` use the same path.

```julia
A = cuNumeric.ones(Float32, 1024, 512)
energy = sum(abs2, A)                     # CNScalar
columns = mapreduce(abs2, +, A; dims=1)   # 1 × 512
const α = 0.5f0                          # a non-const global would infer as Any
distance = sum(x -> abs2(x - α), A)
largest = maximum(abs, A; init=0f0)
finite = all(isfinite, A)                # deferred CNBool
invalid_columns = any(isnan, A; dims=1)  # Bool NDArray, 1 × 512
@allowfetch Bool(finite)                 # convert to a Julia Bool
nonzeros = @allowpromotion count(!iszero, A)  # deferred integer
mean_square = mean(abs2, A; dims=1)      # 1 × 512
```

- Full reductions return a device-resident `CNScalar`.
- Dimensional reductions keep reduced axes with size one. `dims=()` still
  applies the mapping.

## Types and `init`

- `sum` and `prod` widen integers like Base. `mapreduce` with `+` or `*` does
  not. Implicit widening is subject to `allowpromotion`.
- `init` must be a neutral element. It is applied once and sets the output
  type. For `dims` reductions it must not narrow the accumulator.
- Empty inputs follow Base, so arbitrary mappings may need `init`.

### Boolean mapped reductions

`all(f, A; dims=:)` and `any(f, A; dims=:)` require a Bool-valued predicate.
Empty reductions return `true` for `all` and `false` for `any`. Both return
deferred `CNBool` scalars or Bool NDArrays and do not short-circuit.

For `mapreduce(f, &, A)` or `mapreduce(f, |, A)`, use `init=true` or
`init=false`, respectively, to handle empty inputs with arbitrary predicates.

`count(f, A; dims=:, init=0)` counts matches for a Bool-valued predicate;
`count(A)` counts true values in a Bool array. Integer accumulation requires
`@allowpromotion`, as for summing Bool values. `mean(f, A; dims=:)` averages
the mapped values. Both return deferred scalars or NDArrays when using `dims`.

## Limitations

- GPU only. Unmapped reductions still run on CPU.
- One input array. The mapping must be type-stable and GPU-compilable, and may
  capture only immutable scalars. At top level, captured globals must be
  `const`. No custom reducers.
- No complex `minimum`/`maximum`, and no ComplexF64 `prod`.
- Floating-point results are not bitwise reproducible. NaN propagation is kept.

```@autodocs
Modules = [cuNumeric]
Pages = ["ndarray/mapreduce.jl"]
```
