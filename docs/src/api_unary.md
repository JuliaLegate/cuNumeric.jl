# Unary Operations

>[!NOTE]
> Prefer `@.` for multi-op elementwise expressions so every operator is dotted (especially unary negation). This ensures broadcast operations are fused. See [Kernel Fusion](./perf/kernel_fusion.md).


The following unary operations are supported and can be broadcast over `NDArray`:

- `-`, `!`, `~`, `abs`, `acos`, `acosh`, `asin`, `asinh`, `atan`, `atanh`, `cbrt`, `ceil`, `conj`, `cos`, `cosh`, `deg2rad`, `exp`, `exp2`, `expm1`, `floor`, `imag`, `isfinite`, `isinf`, `isnan`, `log`, `log10`, `log1p`, `log2`, `rad2deg`, `real`, `round`, `sign`, `signbit`, `sin`, `sinh`, `sqrt`, `tan`, `tanh`, `trunc`, `^2`, `^-1` or `inv`

## Differences from Base Julia

- The `acosh` function in Julia will error on inputs outside of the domain (`x >= 1`), but cuNumeric.jl will return `NaN`.
- `round.(A)` uses IEEE round-to-nearest-even (the `RINT` kernel), matching Julia `round(x)` / `RoundNearest`. Only that 1-arg path is wired. `digits`, `sigdigits`, and `RoundingMode` are not supported (`round.(A; digits=n)` errors).
- `floor`, `ceil`, `trunc`, and `signbit` are float-only kernels. `round` also supports complex values. Bool and integer inputs are not accepted (Julia's `floor`/`ceil`/`trunc`/`round` on integers are identity).
- `~` is bitwise not on integers. On `Bool` it matches `!` (the invert kernel rejects `Bool`, so we use logical not).
- Full reductions (`sum`, `mean`, `var`, `std`, `argmax`, …) return a 0-d `NDArray`, not a Julia scalar. Use `unwrap` or `A[]` (with `allowscalar`) to read a host value.
- `var` / `std` match Julia / StatsBase sample statistics (`corrected=true`, divisor `n-1`). Complex is not supported.
- `argmax` / `argmin` are 1-d only (matching Base's `Int` return, not `CartesianIndex`). The result is a 0-d `NDArray{Int64}` of the 1-based index. Complex is not supported.

```@autodocs
Modules = [cuNumeric]
Pages = ["ndarray/unary.jl"]
```
