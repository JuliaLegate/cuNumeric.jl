# Binary Operations

>[!NOTE]
> Prefer `@.` for multi-op elementwise expressions so every operator is dotted. Missing a dot silently changes the meaning and can prevent broadcast fusion. See [Kernel Fusion](./perf/kernel_fusion.md).


The following binary operations are supported and can be applied elementwise to pairs of `NDArray` values:

- `+`, `-`, `*`, `/`, `^`, `%`, `<`, `<=`, `>`, `>=`, `==`, `!=`, `&`, `|`, `⊻`, `<<`, `>>`, `atan`, `hypot`, `max`, `min`, `lcm`, `gcd`, `fld`, `mod`, `rem`, `copysign`

## Differences from Base Julia

- `div` / `÷` are not provided. cuPyNumeric floor-divide matches Julia `fld` (toward `-Inf`), not truncated `div`. For example `-7 ÷ 2` is `-3` in Julia and `-4` for `fld`.
- `&`, `|`, `⊻` are integer and `Bool` bitwise ops. `<<` and `>>` are integers excluding `Bool`.
- `copysign` is float-only.

```@autodocs
Modules = [cuNumeric]
Pages = ["ndarray/binary.jl"]
Filter = t -> !(t isa Function && nameof(t) === :mul!)
```
