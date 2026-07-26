# Unary Operations

Prefer `@.` for multi-op elementwise expressions so every operator is dotted (especially unary negation). That keeps the meaning correct and stays friendly to broadcast fusion. See [Kernel Fusion](./perf/kernel_fusion.md).

```julia
y .= @. -a + sin(b)
```

The following unary operations are supported and can be broadcast over `NDArray`:

- `-`, `!`, `abs`, `acos`, `acosh`, `asin`, `asinh`, `atan`, `atanh`, `cbrt`, `conj`, `cos`, `cosh`, `deg2rad`, `exp`, `exp2`, `expm1`, `floor`, `imag`, `isfinite`, `log`, `log10`, `log1p`, `log2`, `rad2deg`, `real`, `sign`, `signbit`, `sin`, `sinh`, `sqrt`, `tan`, `tanh`, `^2`, `^-1` or `inv`

## Differences from Base Julia

- The `acosh` function in Julia will error on inputs outside of the domain (`x >= 1`), but cuNumeric.jl will return `NaN`.

```@autodocs
Modules = [cuNumeric]
Pages = ["ndarray/unary.jl"]
```
