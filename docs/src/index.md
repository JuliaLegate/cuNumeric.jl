```@raw html
<h1>
  <img src="./assets/logo.png" alt="cuNumeric.jl" width="50">
  cuNumeric.jl
</h1>
```

[![Documentation dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://julialegate.github.io/cuNumeric.jl/dev) [![codecov](https://codecov.io/github/julialegate/cuNumeric.jl/branch/main/graph/badge.svg)](https://app.codecov.io/github/JuliaLegate/cuNumeric.jl) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

cuNumeric.jl wraps and extends the [cuPyNumeric](https://github.com/nv-legate/cupynumeric) library from NVIDIA to bring distributed array computing on GPUs and CPUs to Julia. The central type is `NDArray`, which behaves like Julia's `Array` or the `CuArray` from [CUDA.jl](https://github.com/juliagpu/cuda.jl), but executes across multiple GPUs/CPUs. We implement array-level operations on `NDArray` which can be composed into larger programs without the need for explicit MPI calls or writing CUDA kernels.

cuNumeric.jl requires x86 Linux, an NVIDIA GPU, and Julia >= 1.10. If ARM support is of interest open an issue.

### Quick Start

cuNumeric.jl can be installed with the Julia package manager. Activate your preferred environment and then from the Julia REPL run:

```julia
using Pkg
Pkg.add(url = "https://github.com/JuliaLegate/cuNumeric.jl", rev = "main")
```

The first installation can take a while because it includes several large dependencies, such as the CUDA SDK. To use a local cupynumeric build, see [Build Modes](https://julialegate.github.io/cuNumeric.jl/dev/install).

```julia
using cuNumeric
cuNumeric.versioninfo()
```

> [!WARNING]
> Starting more than one instance of cuNumeric.jl can lead to a hard-crash. The default hardware configuration reserves all available resources.

For more details, see [Hardware](https://julialegate.github.io/cuNumeric.jl/dev/configuration/hardware).

### How `NDArray`s work

The semantics of `NDArray` closely mirror Julia's `Array`, and in most cases it is a drop-in replacement. You can use the same constructors (i.e., `zeros`, `ones`, `rand`), broadcasting, slicing, and linear algebra. Under the hood a few details differ from Base, and knowing them can help you write fast code.

**Data may live across many devices.** An `NDArray` is a logical array whose physical buffers can be partitioned over GPUs and CPUs by the Legate runtime. You write ordinary array code and Legate decides where the data lives and how/when it is communicated between devices. As a result, elementwise indexing (i.e. `arr[1]`) is slow (and is prevented by default). Scalar indexing like this forces synchronization and blocks other tasks from executing.

**Slices are views.** Indexing an `NDArray` with ranges returns a view onto the same store, not a copy. That differs from Base Julia, where `A[1:n]` allocates a new `Array`. Mutations through an `NDArray` slice are visible through other aliases of the same data.

**Reductions return arrays, not Julia scalars.** Reductions such as `sum(A)` produce a **0D or 1D** `NDArray` (axis reductions produce a lower-rank `NDArray`), rather than a bare `Float64` / `Float32`. That keeps the Legate task graph asynchronous instead of forcing synchronization to communicate with the Julia runtime. When you need a plain Julia number, call `unwrap`:

```julia
s = sum(A)          # NDArray{T,0}
x = unwrap(s)       # T, e.g. Float32
```

**The Legate runtime builds a DAG asynchronously.** Calling `cuNumeric.zeros` or `A .+ B` records work into Legate's task graph rather than blocking until every GPU kernel finishes. Results are materialized when you need them (for example `println`, `unwrap`, or converting with `Array(A)`). Hiding latency enables performant code.

For API details see [Initialization](https://julialegate.github.io/cuNumeric.jl/dev/api_initialization) and [NDArray Reference](https://julialegate.github.io/cuNumeric.jl/dev/api). For common performance pitfalls, see [Patterns to Avoid](https://julialegate.github.io/cuNumeric.jl/dev/perf/patterns_to_avoid).

### Kernel Fusion

Nested broadcast expressions fuse into a single kernel by default when on GPU. Prefer `@.` for multi-op elementwise code so every operator is dotted and the expression stays completely fused. Even just forgetting the `.` on unary negation (i.e., `y .= -a .+ b .* c`) will result in unfused code. Use the following pattern instead.

```julia
y .= @. -a + b * c
```

See [Kernel Fusion](https://julialegate.github.io/cuNumeric.jl/dev/perf/kernel_fusion) and [Debugging](https://julialegate.github.io/cuNumeric.jl/dev/debugging) for controls and diagnostics.

### The `@accelerate` macro

`@accelerate` fuses eligible GPU broadcasts within and across statements, then releases materialized temporary `NDArray`s after their last use on CPU or GPU. See [The `@accelerate` Macro](https://julialegate.github.io/cuNumeric.jl/dev/perf/reduce_allocations) for usage guidance.

### Benchmarks

Results and reproduction instructions live under [Benchmark Results](https://julialegate.github.io/cuNumeric.jl/dev/benchmarks/results) and [How to Benchmark](https://julialegate.github.io/cuNumeric.jl/dev/benchmarks/howto).

### Try an example

```julia
using cuNumeric

integrand(x) = @. exp(-x^2)

@accelerate function monte_carlo(N, x_max)
    Ω = 2 * x_max
    raw_samples = cuNumeric.rand(N)
    samples = @. Ω * raw_samples - x_max
    return (Ω / N) * sum(integrand(samples))
end

N = 1_000_000
x_max = 10.0f0
estimate = monte_carlo(N, x_max)

println("Monte-Carlo Estimate: $(estimate)")
```
More worked examples (initialization, Gray-Scott, …) are in the documentation sidebar under **Examples**.

### Known Limitations

- There is no support for `Float16` or `ComplexF16`
