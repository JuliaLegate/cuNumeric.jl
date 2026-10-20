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
`prod`, `minimum`, `maximum`, `all`, and `any` use the same path.

```julia
A = cuNumeric.ones(Float32, 1024, 512)
energy = sum(abs2, A)                     # CNScalar
columns = mapreduce(abs2, +, A; dims=1)   # 1 × 512
const α = 0.5f0                          # a non-const global would infer as Any
distance = sum(x -> abs2(x - α), A)
largest = maximum(abs, A; init=0f0)
finite = mapreduce(isfinite, &, A; init=true)  # deferred CNBool
invalid_columns = mapreduce(isnan, |, A; dims=1)  # Bool NDArray, 1 × 512
finite = all(isfinite, A)                # equivalent predicate API
invalid_columns = any(isnan, A; dims=1)
@allowfetch Bool(finite)                  # explicit host-control boundary
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

`&` and `|` accept any supported Bool-valued GPU mapping, including predicates,
`identity` on Bool arrays, and closures with scalar captures. They use the same
fused reduction kernels as numeric operators; no predicate array is allocated.
These operators do not short-circuit the mapping.

`all(f, A; dims=:)` and `any(f, A; dims=:)` use these fused kernels with
`init=true` and `init=false`, respectively. Empty reductions therefore return
true for `all` and false for `any`, including arbitrary supported predicates.
They accept `dims`, but not an `init` keyword, and require Bool-valued predicates.
Like Boolean `mapreduce`, they return deferred results and do not short-circuit.

Use `true` as the identity for `&`, and `false` for `|`. Dimensional reductions
without `init` use these identities, including empty reduced axes. Full empty
reductions follow Base's `mapreduce_empty`: for example, `identity` on an empty
Bool array has an identity, while an arbitrary predicate may require explicit
`init`. `dims=()` applies the mapping and seed without reducing any axes.

Integer `init` values follow Base's bitwise result typing, subject to the usual
promotion checks. The mapped values must still be Bool; this is not support for
reducing integer bit patterns. Bool results stay device-resident: zero-dimensional
results (including full reductions) are `CNBool`, and higher-dimensional results
are Bool NDArrays.

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
