# Binary Operations

Prefer `@.` for multi-op elementwise expressions so every operator is dotted. Missing a dot (especially on unary negation or addition) silently changes the meaning and can prevent broadcast fusion. See [Kernel Fusion](./perf/kernel_fusion.md).

```julia
y .= @. -a + b * c
```

The following binary operations are supported and can be applied elementwise to pairs of `NDArray` values:

- `+`, `-`, `*`, `/`, `^`, `<`, `<=`, `>`, `>=`, `==`, `!=`, `atan`, `hypot`, `max`, `min`, `lcm`, `gcd`

For 1D arrays and scalars, `*` is elementwise (or scaling). For two 2D arrays, `*` is matrix multiplication. Use `.*` when you want an elementwise product of matrices. See [Linear Algebra](./linalg.md) for `*` / `mul!`.

```@autodocs
Modules = [cuNumeric]
Pages = ["ndarray/binary.jl"]
Filter = t -> !(t isa Function && nameof(t) === :mul!)
```
