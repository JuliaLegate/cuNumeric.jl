# Binary Operations

>[!NOTE]
> Prefer `@.` for multi-op elementwise expressions so every operator is dotted. Missing a dot silently changes the meaning and can prevent broadcast fusion. See [Kernel Fusion](./perf/kernel_fusion.md).


The following binary operations are supported and can be applied elementwise to pairs of `NDArray` values:

- `+`, `-`, `*`, `/`, `^`, `%`, `<`, `<=`, `>`, `>=`, `==`, `!=`, `&`, `|`, `⊻`, `<<`, `>>`, `atan`, `hypot`, `max`, `min`, `lcm`, `gcd`, `fld`, `mod`, `rem`, `copysign`

## Differences from Base Julia

- `div` / `÷` are not provided. cuPyNumeric floor-divide matches Julia `fld` (toward `-Inf`), not truncated `div`. For example `-7 ÷ 2` is `-3` in Julia and `-4` for `fld`.
- `&`, `|`, `⊻` are integer and `Bool` bitwise ops. `<<` and `>>` are integers excluding `Bool`.
- `copysign` is float-only.

## Elementwise select

`cuNumeric.where(cond, x, y)` takes `x` where the `NDArray{Bool}` `cond` is true
and `y` everywhere else. `ifelse(cond, x, y)` is the same call. Either branch may
be a scalar, and all three operands broadcast against each other.

```julia
mask = arr .> 0.0f0
clipped = cuNumeric.where(mask, arr, 0.0f0)
same = ifelse(mask, arr, 0.0f0)
```

The branches are promoted to a common element type under the usual rules, so
widening one of them needs `@allowpromotion`.

Only the whole-array call is supported. `ifelse.(cond, x, y)` is not, because
broadcast promotes every operand to a single element type, which would strip
`Bool` off the condition.

To find the positions the mask selected rather than the values, use `findall`
(see [Unary Operations](./api_unary.md)).

```@autodocs
Modules = [cuNumeric]
Pages = ["ndarray/binary.jl"]
Filter = t -> !(t isa Function && nameof(t) === :mul!)
```
