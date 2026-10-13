# Unary Operations

>[!NOTE]
> Prefer `@.` for multi-op elementwise expressions so every operator is dotted (especially unary negation). This makes eligible CUDA broadcasts fusion-friendly. See [Kernel Fusion](./perf/kernel_fusion.md).


The following unary operations are supported and can be broadcast over `NDArray`:

- `-`, `!`, `~`, `abs`, `acos`, `acosh`, `asin`, `asinh`, `atan`, `atanh`, `cbrt`, `ceil`, `conj`, `cos`, `cosh`, `deg2rad`, `exp`, `exp2`, `expm1`, `floor`, `imag`, `isfinite`, `isinf`, `isnan`, `log`, `log10`, `log1p`, `log2`, `rad2deg`, `real`, `round`, `sign`, `signbit`, `sin`, `sinh`, `sqrt`, `tan`, `tanh`, `trunc`, `^2`, `^-1` or `inv`

Reductions, with optional `dims` except `argmax` / `argmin`:

- `sum`, `prod`, `maximum`, `minimum`, `mean`, `var`, `std`, `all`, `any`, `argmax`, `argmin`

## Differences from Base Julia

- The `acosh` function in Julia will error on inputs outside of the domain (`x >= 1`), but cuNumeric.jl will return `NaN`.
- `round.(A)` uses IEEE round-to-nearest-even (the `RINT` kernel), matching Julia `round(x)` / `RoundNearest`. Only the single-argument form is supported. `digits`, `sigdigits`, and `RoundingMode` are not supported (`round.(A; digits=n)` errors).
- `floor`, `ceil`, `trunc`, and `signbit` are float-only kernels. `round` also supports complex values. Bool and integer inputs are not accepted (Julia's `floor`/`ceil`/`trunc`/`round` return integer inputs unchanged).
- `~` is bitwise not on integers. On `Bool` it matches `!` (the invert kernel rejects `Bool`, so we use logical not).
- Full reductions (`sum`, `mean`, `var`, `std`, …) return a `CNScalar` backed by a 0D NDArray. Use `fetch` to read a native Julia scalar.
- `var` / `std` match Julia / StatsBase sample statistics (`corrected=true`, divisor `n-1`). Complex inputs are not supported.
- `argmax` / `argmin` are 1D fetch (matching Base's `Int` return, not `CartesianIndex`). The result is a 0D `NDArray{Int64}` of the 1-based index. Complex inputs are not supported.

For `mapreduce` and mapped `sum`, `prod`, `minimum`, and `maximum`, see
[Map and Reduce](./api_mapreduce.md).

```@autodocs
Modules = [cuNumeric]
Pages = ["ndarray/unary.jl"]
Filter = t -> !(t isa Function && nameof(t) === :map)
```
