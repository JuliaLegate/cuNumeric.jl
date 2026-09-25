# Mapped Reductions

`mapreduce(f, op, A; dims=:, init=...)` maps and reduces one NDArray without
allocating a mapped copy. Supported operators are `+`, `*`, `min`, and `max`.
Mapped `sum`, `prod`, `minimum`, and `maximum` use the same implementation.

```julia
A = cuNumeric.ones(Float32, 1024, 512)
energy = sum(abs2, A)                         # CNScalar
columns = mapreduce(abs2, +, A; dims=1)        # 1 × 512
α = 0.5f0
distance = sum(x -> abs2(x - α), A)
largest = maximum(abs, A; init=0f0)
```

Full reductions return a device-resident CNScalar. Dimensional reductions
keep reduced axes with size one. Duplicate dimensions are ignored; positive
out-of-rank dimensions have no effect. `dims=()` still applies the mapping.

## Types and initialization

`sum` and `prod` use Base's integer widening rules; `mapreduce` with `+` or `*`
uses ordinary scalar arithmetic. Implicit widening, including within the mapping,
is subject to `allowpromotion`.

Supply a neutral scalar `init`; it is applied once. Full empty reductions return
`init` when supplied. Otherwise, Base's empty-input rules apply: dimensional
sums/products return their identities, while empty extrema error except for
special cases such as `maximum(abs2, A)`. Arbitrary mappings may require `init`.

For full reductions, combining with `init` determines the output type. For
dimensional reductions, `typeof(init)` determines it and must accommodate the
accumulator without narrowing.

## Current limitations

- GPU execution only, independent of broadcast-fusion settings. Existing unmapped
  reductions retain CPU support. Participating GPUs must support the compilation
  target; heterogeneous target selection is unsupported.
- One input NDArray and a type-stable, GPU-compilable mapping with immutable scalar
  captures. Captured arrays/pointers, host allocation or side effects, and custom
  reducers are unsupported.
- Results may be Bool, the supported integer widths, Float32/Float64, or
  ComplexF32/ComplexF64. Complex extrema and ComplexF64 product accumulators
  (including those selected by dimensional `init`) are unsupported.
- Floating-point reassociation can change rounding, overflow, and arithmetic
  signed zeros. Bitwise reproducibility is not guaranteed. Floating-point extrema
  preserve NaN propagation and signed-zero ordering, but not NaN payloads.

```@autodocs
Modules = [cuNumeric]
Pages = ["ndarray/mapreduce.jl"]
```
