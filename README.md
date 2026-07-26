# cuNumeric.jl

[Documentation dev](https://julialegate.github.io/cuNumeric.jl/dev/) [codecov](https://app.codecov.io/github/JuliaLegate/cuNumeric.jl) [License: MIT](https://opensource.org/licenses/MIT)

The cuNumeric.jl package wraps the [cuPyNumeric](https://github.com/nv-legate/cupynumeric) C++ API from NVIDIA to bring distributed array computing on GPUs and CPUs to Julia. The central type is `NDArray`, which behaves like Julia's `Array` or CUDA.jl's `CuArray`, but executes across multiple GPUs. We implement array-level operations on `NDArray` which can be composed into larger programs without the need for explicit MPI calls or writing CUDA kernels.

cuNumeric.jl requires x86 Linux, an NVIDIA GPU, and Julia >= 1.10. If ARM support is of interest open an issue.

### Quick Start

cuNumeric.jl can be installed with the Julia package manager. Active your preferred environment and then from the Julia REPL run:

```julia
using Pkg
Pkg.add(url = "https://github.com/JuliaLegate/cuNumeric.jl", rev = "main")
```

The first run might take awhile as it has to install multiple large dependencies such as the CUDA SDK (if you have an NVIDIA GPU). To use a local build of cupynumeric.so, see [Build Modes](./install.md).

```julia
using cuNumeric
cuNumeric.versioninfo()
```

> [!WARNING]
> Starting more than one instance of cuNumeric.jl can lead to a hard-crash. The default hardware configuration reserves all available resources.

For more details, see [Hardware](./configuration/hardware.md).

### How `NDArray`s work

The semantics of `NDArray` closely mirror Julia's `Array`, and in most cases it is a drop-in replacement: the same constructors (`zeros`, `ones`, `rand`, …), broadcasting with `.` / `@.`, slicing, and linear algebra. Under the hood a few details differ from Base, and knowing them can help you write fast code.

**Data may live across many devices.** An `NDArray` is a logical array whose physical buffers can be partitioned over GPUs and CPUs by the Legate runtime. You write ordinary array code and Legate decides where tiles live and moves or communicates data when an operation needs it. Elementwise looping from the host is the slow path (and is blocked by default). It forces synchronization and can pull distributed data back to a single process.

**Slices are views.** Indexing an `NDArray` with ranges returns a view onto the same store, not a copy. That differs from Base Julia, where `A[1:n]` on an `Array` allocates a new array (use `@view` there for a view). Mutations through an `NDArray` slice are visible through other aliases of the same data. Assignments into slices still go through the Legate task system.

**Reductions return arrays, not Julia scalars.** Full reductions such as `sum(A)` produce a **0D** `NDArray` (axis reductions produce a lower-rank `NDArray`), rather than a bare `Float64` / `Float32`. That keeps the Legate task graph asynchronous instead of forcing an immediate host sync. When you need a plain Julia number, call `unwrap`:

```julia
s = sum(A)          # NDArray{T,0}
x = unwrap(s)       # T, e.g. Float32
```

**The Legate runtime builds a DAG asynchronously.** Calling `cuNumeric.zeros` or `A .+ B` typically records work into Legate's task graph rather than blocking until every GPU kernel finishes. Results are materialized when you need them (for example printing, `unwrap`, converting with `Array(A)`, or explicit synchronization). Hiding latency enables performant code.

For API details see [Initialization](./api_initialization.md) and [NDArray Reference](./api.md). For anti-patterns that kill performance, see [Patterns to Avoid](./perf/patterns_to_avoid.md).

### Kernel Fusion

Nested broadcast expressions fuse into a single kernel by default when on GPU. Prefer `@.` for multi-op elementwise code so every operator is dotted and the expression stays completely fused:

```julia
y .= @. -a + b * c
```

See [Kernel Fusion](./perf/kernel_fusion.md) and [Debugging](./debugging.md) for controls and printers.

### Helping the Garbage Collector

Many calls such as array slicing and un-fused broadcasts allocate a new `NDArray`. The legate runtime keeps track of all references to the underlying data and will not free the memory until Julia's GC frees the `NDArray` handles. Because Julia's GC runs on memory pressure and an `NDArray` only stores a pointer (i.e., Julia's GC does not know the true size), many dead buffers accumulate and can cause out-of-memory errors.

`@analyze_lifetimes` performs a **static last-use analysis** at macro-expansion time and inserts eager calls to immediately delete un-needed temporary `NDArrays`. These buffers can then be reused by legate later for same-sized allocations.

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
