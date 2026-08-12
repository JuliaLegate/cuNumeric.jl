# NDArray Reference

Indexing, reshaping, reductions, comparisons, memory helpers, lifetime macros, and related utilities. For constructors (`zeros`, `ones`, `rand`, …) see [Initialization](./api_initialization.md).

```@autodocs
Modules = [cuNumeric]
Pages = ["ndarray/ndarray.jl", "ndarray/linalg.jl", "cuNumeric.jl", "warnings.jl", "util.jl", "memory.jl", "scoping/scoping.jl"]
Filter = t -> !(t isa Function && nameof(t) in (:zeros, :ones, :fill, :trues, :falses, :eye, :rand, :rand!))
```
