```@raw html
<p style="text-align: center;">
  <img src="./assets/logo-horizontal.png" alt="cuNumeric.jl" width="800" style="max-width: 100%; height: auto; margin: 0 auto;">
</p>
```

[![Documentation stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://julialegate.github.io/cuNumeric.jl/stable) [![Documentation dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://julialegate.github.io/cuNumeric.jl/dev) [![codecov](https://codecov.io/github/julialegate/cuNumeric.jl/branch/main/graph/badge.svg)](https://app.codecov.io/github/JuliaLegate/cuNumeric.jl) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

cuNumeric.jl wraps and extends the [cuPyNumeric](https://github.com/nv-legate/cupynumeric) library from NVIDIA to bring distributed array computing on GPUs and CPUs to Julia. The central type is `NDArray`, which behaves like Julia's `Array` or the `CuArray` from [CUDA.jl](https://github.com/juliagpu/cuda.jl), but executes across multiple GPUs/CPUs. We implement array-level operations on `NDArray` which can be composed into larger programs without the need for explicit MPI calls or writing CUDA kernels.

cuNumeric.jl requires x86 Linux, an NVIDIA GPU, and Julia >= 1.10. If you need ARM support, please open an issue.

### Quick Start

cuNumeric.jl can be installed with the Julia package manager. Activate your preferred environment, then run the following in the Julia REPL:

```julia
using Pkg
Pkg.add("cuNumeric")
```

The first installation can take a while because it includes several large dependencies, such as the CUDA SDK.

`````@raw html
<!--
To use a local cuPyNumeric build, see [Build Modes](install.md).
-->
`````


```julia
using cuNumeric
cuNumeric.versioninfo()
```

> [!WARNING]
> Starting more than one instance of cuNumeric.jl can lead to a crash. The default hardware configuration reserves all available resources.

For more details, see [Hardware](configuration/hardware.md).

### How `NDArray`s work

`NDArray` behaves much like Julia's `Array`, with familiar constructors (e.g., `zeros`, `ones`, `rand`), broadcasting, slicing, and linear algebra. A few differences matter for correctness and performance.

**`NDArray <: AbstractArray`.** This lets generic Julia code work with `NDArray`,
including the examples using Krylov.jl and OrdinaryDiffEq.jl. Some behaviors,
such as scalar indexing, are disabled by default. See [Patterns to Avoid](./perf/patterns_to_avoid.md)
for macros such as `@allowscalar` that selectively enable them.

**Data may live across many devices.** An `NDArray` is a logical array whose physical buffers can be partitioned over GPUs and CPUs by the runtime. You write ordinary array code and the runtime decides where the data lives and when data is transferred between devices. As a result, scalar indexing (e.g., `arr[1]`) is slow and disabled by default. Scalar indexing like this forces synchronization and blocks other tasks from executing.

**Slices are views.** Indexing an `NDArray` with ranges returns a view onto the same store, not a copy. That differs from Base Julia, where `A[1:n]` allocates a new `Array`. Mutations through an `NDArray` slice are visible through other aliases of the same data.

**Scalar reductions return device scalars.** Full reductions such as `sum(A)` return a `CNScalar` backed by a 0D NDArray. Reductions that retain dimensions still return NDArrays. This keeps the task graph asynchronous. When you need a native Julia scalar, call `fetch`:

```julia
s = sum(A)          # CNScalar
x = fetch(s)        # native Julia scalar, e.g. Float32
```

**The Legate runtime builds a DAG asynchronously.** Calling `cuNumeric.zeros` or `A .+ B` records work into a task graph rather than blocking until every GPU kernel finishes. Results are materialized when you need them (for example `println`, `fetch`, or converting with `Array(A)`).

For API details see [Initialization](https://julialegate.github.io/cuNumeric.jl/stable/api_initialization) and [NDArray Reference](https://julialegate.github.io/cuNumeric.jl/stable/api). For common performance pitfalls, see [Patterns to Avoid](./perf/patterns_to_avoid.md).

### Kernel Fusion

Nested broadcast expressions fuse into a single kernel by default on a GPU.
Use `@.` for elementwise expressions with multiple operations so every operator
is dotted. A missing dot, even on unary negation (`y .= -a .+ b .* c`),
prevents full fusion. Write this as:

```julia
y .= @. -a + b * c
```

See [Kernel Fusion](perf/kernel_fusion.md) and [Debugging](debugging.md) for controls and diagnostics.

### The `@accelerate` macro

`@accelerate` fuses eligible GPU broadcasts within and across statements, then releases materialized temporary `NDArray`s after their last use on CPU or GPU. See [The `@accelerate` Macro](perf/accelerate.md) for usage guidance.

### Benchmarks

See [Benchmarks](benchmarks/results.md) for package integration results and reproduction instructions.

### Try an example

```julia
using cuNumeric

integrand(x) = exp(-x^2)

@accelerate function monte_carlo(N, x_max)
    Ω = 2 * x_max
    raw_samples = cuNumeric.rand(N)
    samples = @. Ω * raw_samples - x_max
    return (Ω / N) * sum(integrand.(samples))
end

N = 1_000_000
x_max = 10.0f0
estimate = monte_carlo(N, x_max)

println("Monte-Carlo Estimate: $(fetch(estimate))")
```
More worked examples (initialization, Gray-Scott, …) are in the documentation sidebar under **Examples**.

### Known Limitations

- There is no support for `Float16` or `ComplexF16`
- Stochastic bug in FFT can cause segfault
- Unfused broadcasting of a `Vector` reshaped to (1, n) will not mimic Julia semantics
