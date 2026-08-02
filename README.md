<h1>
  <img src="docs/src/assets/logo.png" alt="cuNumeric.jl" width="50">
  <a href="https://julialegate.github.io/cuNumeric.jl/dev/">cuNumeric.jl</a>
</h1>

[![Documentation dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://julialegate.github.io/cuNumeric.jl/dev/) [![codecov](https://codecov.io/github/julialegate/cuNumeric.jl/branch/main/graph/badge.svg)](https://app.codecov.io/github/JuliaLegate/cuNumeric.jl) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

cuNumeric.jl wraps and extends the [cuPyNumeric](https://github.com/nv-legate/cupynumeric) library from NVIDIA to bring distributed array computing on GPUs and CPUs to Julia. The central type is `NDArray`, which behaves like Julia's `Array` or the `CuArray` from [CUDA.jl](https://github.com/juliagpu/cuda.jl), but executes across multiple GPUs/CPUs. We implement array-level operations on `NDArray` which can be composed into larger programs without the need for explicit MPI calls or writing CUDA kernels.

cuNumeric.jl requires x86 Linux, an NVIDIA GPU, and Julia >= 1.10. If ARM support is of interest open an issue.

### Quick Start

cuNumeric.jl can be installed with the Julia package manager. Activate your preferred environment and then from the Julia REPL run:

```julia
using Pkg
Pkg.add(url = "https://github.com/JuliaLegate/cuNumeric.jl", rev = "main")
```

The first time might take awhile as it has to install multiple large dependencies such as the CUDA SDK (if you have an NVIDIA GPU). To use a local build of cupynumeric.so, see [Build Modes](./install.md).

```julia
using cuNumeric
cuNumeric.versioninfo()
```

> [!WARNING]
> Starting more than one instance of cuNumeric.jl can lead to a hard-crash. The default hardware configuration reserves all available resources.

For more details, see [Hardware](./configuration/hardware.md).

### How `NDArray`s work

The semantics of `NDArray` closely mirror Julia's `Array`, and in most cases it is a drop-in replacement. You can use the same constructors (i.e., `zeros`, `ones`, `rand`), broadcasting, slicing, and linear algebra. Under the hood a few details differ from Base, and knowing them can help you write fast code.

**Data may live across many devices.** An `NDArray` is a logical array whose physical buffers can be partitioned over GPUs and CPUs by the Legate runtime. You write ordinary array code and Legate decides where the data lives and how/when it is communicated between devices. As a result, elementwise indexing (i.e. `arr[1]`) is slow (and is prevented by default). Scalar indexing like this forces synchronization and blocks other tasks from executing. Functions like `println` result in data being copied to the host and can also be slow.

**Slices are views.** Indexing an `NDArray` with ranges returns a view onto the same store, not a copy. That differs from Base Julia, where `A[1:n]` allocates a new `Array`. Mutations through an `NDArray` slice are visible through other aliases of the same data.

**Reductions return arrays, not Julia scalars.** Reductions such as `sum(A)` produce a **0D or 1D** `NDArray` (axis reductions produce a lower-rank `NDArray`), rather than a bare `Float64` / `Float32`. That keeps the Legate task graph asynchronous instead of forcing synchronization to communite with the Julia runtime. When you need a plain Julia number, call `unwrap` or `only`:

```julia
s = sum(A)          # NDArray{T,0}
x = unwrap(s)       # T, e.g. Float32
x2 = only(s)
```

**The Legate runtime builds a DAG asynchronously.** Calling `cuNumeric.zeros` or `A .+ B` records work into Legate's task graph rather than blocking until every GPU kernel finishes. Results are materialized when you need them (for example `println`, `unwrap`, or converting with `Array(A)`). Hiding latency enables performant code.

For API details see [Initialization](./api_initialization.md) and [NDArray Reference](./api.md). For anti-patterns that kill performance, see [Patterns to Avoid](./perf/patterns_to_avoid.md).

### Kernel Fusion

Nested broadcast expressions fuse into a single kernel by default when on GPU. Prefer `@.` for multi-op elementwise code so every operator is dotted and the expression stays completely fused. Even just forgetting the `.` on unary negation (i.e., `y .= -a .+ b .* c`) will result in unfused code. Use the following pattern instead.

```julia
y .= @. -a + b * c
```

See [Kernel Fusion](./perf/kernel_fusion.md) and [Debugging](./debugging.md) for controls and pretty printers.

### Helping the Garbage Collector

Many calls such as array slicing and un-fused broadcasts allocate a new `NDArray`. The Legate runtime keeps track of all references to the underlying data and will not free the memory until Julia's GC frees the `NDArray` handles. Because Julia's GC runs on memory pressure and an `NDArray` only stores a pointer (i.e., Julia's GC does not know the true size), many dead buffers accumulate and can cause out-of-memory errors.

`@analyze_lifetimes` performs a **static last-use analysis** at macro-expansion time and inserts eager calls to immediately free unused `NDArrays`. These buffers can then be reused by legate later for same-sized allocations.

```julia
@analyze_lifetimes begin
    result = @. A[1:end, :] + B[1:end, :]
    C .= @. result * 2.0f0
end
```

### Performance at a glance

A representative benchmark figure will go here (add something like `docs/src/images/benchmarks-overview.png` when ready).

Numbers, plots, and how to reproduce them live under [Benchmark Results](./benchmarks/results.md) and [How to Benchmark](./benchmarks/howto.md).

### Try an example

```julia
using cuNumeric

integrand = (x) -> @. exp(-x^2)

N = 1_000_000
x_max = 10.0f0
Ω = 2 * x_max

samples = Ω .* cuNumeric.rand(N)
samples = samples .- x_max
estimate = (Ω / N) .* sum(integrand(samples))

println("Monte-Carlo Estimate: $(estimate)")
```
More worked examples (initialization, Gray-Scott, …) are in the documentation sidebar under **Examples**.

### Known Limitations

- There is no support for `Float16` or `ComplexF16`
