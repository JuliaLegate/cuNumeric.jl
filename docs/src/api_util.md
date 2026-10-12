# Indexing, Sorting & Utilities

## Indexing and slices

1D and 2D arrays support slicing with `:`, unit ranges (`2:4`) and integers.
A slice is a view: writing to it changes the original array.

```julia
A = cuNumeric.rand(Float32, 100, 100)
top = A[1:10, :]               # view, no copy
A[2:3, :] = B                  # copy B into rows 2–3
A[:, 5] .= 0f0                 # fill a column
A[2:3, :] .= A[1:2, :] .+ 1f0  # broadcast into a slice
```

Differences from Base:

- A slice with an integer index keeps that dimension: `A[:, 2]` is `n×1` and
  `A[2, :]` is `1×n`, not a vector.
- Assign a scalar to a slice with `.=`. `A[1:2, :] = 0f0` errors.
- Not supported: step ranges (`1:2:n`), index or Boolean arrays, and slicing
  arrays with 3 or more dimensions.
- Single elements (`A[2, 3]`) need [`@allowscalar`](./perf/patterns_to_avoid.md).

## Sorting and searching

`sort`, `sort!`, `unique`, `searchsortedfirst`, `searchsortedlast` and
`searchsorted` are not Base's: call them with the `cuNumeric.` prefix.

```julia
v = cuNumeric.rand(Float32, 1000)
s = cuNumeric.sort(v)                     # cuNumeric.sort!(v) sorts in place
i = cuNumeric.searchsortedfirst(s, 0.5f0) # 0D NDArray; fetch(i) for an Int
u = cuNumeric.unique(NDArray([3, 1, 3]))  # [1, 3], sorted
```

N-d arrays need `dims`. Sorts are unstable unless `stable=true`.

```@autodocs
Modules = [cuNumeric]
Pages = ["ndarray/ndarray.jl", "ndarray/sort.jl", "cuNumeric.jl", "warnings.jl", "util.jl", "memory.jl", "scoping/scoping.jl", "scoping/accelerate.jl"]
Filter = t -> !(t isa Function && nameof(t) in (:zeros, :ones, :fill, :trues, :falses, :eye, :rand, :rand!, :randn, :randn!, :randexp, :randexp!, :default_rng, :random, :random!))
```
