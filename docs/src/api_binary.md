# Binary Operations

>[!NOTE]
> Prefer `@.` for multi-op elementwise expressions so every operator is dotted. Missing a dot silently changes the meaning and can prevent broadcast fusion. See [Kernel Fusion](./perf/kernel_fusion.md).


The following binary operations are supported and can be applied elementwise to pairs of `NDArray` values:

- `+`, `-`, `*`, `/`, `^`, `<`, `<=`, `>`, `>=`, `==`, `!=`, `atan`, `hypot`, `max`, `min`, `lcm`, `gcd`

```@autodocs
Modules = [cuNumeric]
Pages = ["ndarray/binary.jl"]
Filter = t -> !(t isa Function && nameof(t) === :mul!)
```
