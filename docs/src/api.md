# NDArray Reference

Indexing, reshaping, reductions, comparisons, memory helpers, acceleration macros, and related utilities. For constructors (`zeros`, `ones`, `rand`, …) see [Initialization](./api_initialization.md). For RNG engines and `default_rng`, see [Random](./api_random.md).

```@autodocs
Modules = [cuNumeric]
Pages = ["ndarray/ndarray.jl", "ndarray/linalg.jl", "ndarray/batched_linalg.jl", "cuNumeric.jl", "warnings.jl", "util.jl", "memory.jl", "scoping/scoping.jl", "scoping/accelerate.jl"]
Filter = t -> !(t isa Function && nameof(t) in (:zeros, :ones, :fill, :trues, :falses, :eye, :rand, :rand!, :randn, :randn!, :randexp, :randexp!, :default_rng, :random, :random!))
```
