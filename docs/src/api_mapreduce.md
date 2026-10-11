# Map and Mapped Reductions

## `map` and `map!`

`map(f, A)` and `map(f, A, B)` are the same as `f.(A)` and `f.(A, B)`.
`map!(op, dest, A, B)` writes into `dest` for built-in binary operators
(`+`, `*`, `max`, …); all three arrays must have the same shape.

```julia
A, B = cuNumeric.rand(1024), cuNumeric.rand(1024)
C = map(sin, A)
map!(*, C, A, B)
```

## `mapreduce`

`mapreduce(f, op, A; dims=:, init=...)` maps and reduces an NDArray in one
pass, without a mapped copy. `op` can be `+`, `*`, `min` or `max`. Mapped `sum`,
`prod`, `minimum` and `maximum` use the same path.

```julia
A = cuNumeric.ones(Float32, 1024, 512)
energy = sum(abs2, A)                     # CNScalar
columns = mapreduce(abs2, +, A; dims=1)   # 1 × 512
α = 0.5f0
distance = sum(x -> abs2(x - α), A)
largest = maximum(abs, A; init=0f0)
```

- Full reductions return a device-resident `CNScalar`.
- Dimensional reductions keep reduced axes with size one. `dims=()` still
  applies the mapping.

## Types and `init`

- `sum` and `prod` widen integers like Base; `mapreduce` with `+` or `*` does
  not. Implicit widening is subject to `allowpromotion`.
- `init` must be a neutral element; it is applied once and sets the output
  type. For `dims` reductions it must not narrow the accumulator.
- Empty inputs follow Base, so arbitrary mappings may need `init`.

## Limitations

- GPU only. Unmapped reductions still run on CPU.
- One input array. The mapping must be type-stable and GPU-compilable, and may
  capture only immutable scalars. No custom reducers.
- No complex `minimum`/`maximum`, and no ComplexF64 `prod`.
- Floating-point results are not bitwise reproducible. NaN propagation is kept.

```@autodocs
Modules = [cuNumeric]
Pages = ["ndarray/mapreduce.jl"]
```
