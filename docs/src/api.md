# NDArray Reference

Indexing, reshaping, reductions, comparisons, memory helpers, lifetime macros, and related utilities. For constructors (`zeros`, `ones`, `rand`, …) see [Initialization](./api_initialization.md). For RNG engines and `default_rng`, see [Random](./api_random.md). For `fft` / `ifft` / `fft!` / `ifft!` and `batched_fft`, see [FFT](./fft.md). There is no `plan_fft`: cupynumeric does not expose a cuFFT handle.

For `mapreduce` and the mapped forms of `sum`, `prod`, `minimum`, and `maximum`,
see [Map and Reduce](./api_mapreduce.md).

## Sorting and searching

`sort`, `sort!`, `unique`, `searchsortedfirst`, `searchsortedlast` and
`searchsorted` are not Base's: call them with the `cuNumeric.` prefix.

```julia
v = cuNumeric.rand(Float32, 1000)
s = cuNumeric.sort(v)                     # cuNumeric.sort!(v) sorts in place
i = cuNumeric.searchsortedfirst(s, 0.5f0) # 0-d NDArray; fetch(i) for an Int
u = cuNumeric.unique(NDArray([3, 1, 3]))  # [1, 3], sorted
```

N-d arrays need `dims`. Sorts are unstable unless `stable=true`.

```@autodocs
Modules = [cuNumeric]
Pages = ["ndarray/ndarray.jl", "ndarray/linalg.jl", "ndarray/batched_linalg.jl", "ndarray/sort.jl", "cuNumeric.jl", "warnings.jl", "util.jl", "memory.jl", "scoping/scoping.jl", "scoping/accelerate.jl"]
Filter = t -> !(t isa Function && nameof(t) in (:zeros, :ones, :fill, :trues, :falses, :eye, :rand, :rand!, :randn, :randn!, :randexp, :randexp!, :default_rng, :random, :random!))
```
