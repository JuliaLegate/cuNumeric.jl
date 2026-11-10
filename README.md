<p align="center">
  <a href="https://julialegate.github.io/cuNumeric.jl/stable">
    <img src="docs/src/assets/logo-horizontal.png" alt="cuNumeric.jl" width="800">
  </a>
</p>

[![Documentation stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://julialegate.github.io/cuNumeric.jl/stable) [![codecov](https://codecov.io/github/julialegate/cuNumeric.jl/branch/main/graph/badge.svg)](https://app.codecov.io/github/JuliaLegate/cuNumeric.jl) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

cuNumeric.jl wraps and extends the [cuPyNumeric](https://github.com/nv-legate/cupynumeric) library from NVIDIA to bring distributed array computing on GPUs and CPUs to Julia. The central type is `NDArray`, which behaves like Julia's `Array` or the `CuArray` from [CUDA.jl](https://github.com/juliagpu/cuda.jl), but executes across multiple GPUs/CPUs. We implement array-level operations on `NDArray` which can be composed into larger programs without the need for explicit MPI calls or writing CUDA kernels.

cuNumeric.jl requires x86 Linux, an NVIDIA GPU, and Julia >= 1.10. If ARM support is of interest open an issue.

### Quick Start

cuNumeric.jl can be installed with the Julia package manager. Activate your preferred environment and then from the Julia REPL run:

```julia
using Pkg
Pkg.add("cuNumeric")
```

The first installation can take a while because it includes several large dependencies, such as the CUDA SDK. After successful installation you can check the install details by calling `versioninfo`.
```julia
using cuNumeric
cuNumeric.versioninfo()
```

> [!WARNING]
> **The default hardware configuration reserves all available resources.** Starting more than one instance of cuNumeric.jl or starting cuNumeric.jl on a GPU with existing allocations can lead to a hard-crash. For more details, see [Hardware](https://julialegate.github.io/cuNumeric.jl/stable/configuration/hardware).

### How `NDArray`s work

The semantics of `NDArray` closely mirror Julia's `Array`, and in most cases it is a drop-in replacement. You can use the same constructors (i.e., `zeros`, `ones`, `rand`), broadcasting, slicing, and linear algebra. Under the hood a few details differ from Base, and knowing them can help you write fast code.

**Data may live across many devices.** An `NDArray` is a logical array whose physical buffers can be stored across multiple GPUs and CPUs by the Legate runtime. You write ordinary array code and Legate decides where the data lives and how/when it is communicated between devices. As a result, elementwise indexing (i.e. `arr[1]`) is slow (and is prevented by default). Scalar indexing like this forces synchronization and blocks other tasks from executing.

**Slices are views.** Indexing an `NDArray` with ranges returns a view onto the same store, not a copy. That differs from Base Julia, where `A[1:n]` allocates a new `Array`. Mutating an `NDArray` slice mutates the parent and all other aliases of the underlying data.

**Scalar reductions return device scalars.** Full reductions such as `sum(A)`, `dot(x,y)`, and `norm(x)` return numeric wrappers (`CNFloat`, `CNInt`, `CNUInt`, `CNBool`, or `CNComplex`) backed by 0D NDArrays. Each subtypes the corresponding abstract numeric category. Dimension-preserving reductions still return NDArrays. Use `fetch` for explicit host extraction:

```julia
s = sum(A)          # CNScalar
x = fetch(s)        # native Julia scalar, e.g. Float32
```

Use `allowfetch() do ... end` or `@allowfetch` to permit host comparisons and numeric conversions of device scalars. See [Device scalars](https://julialegate.github.io/cuNumeric.jl/stable/api_cnscalar).

**Underlying data is row-major** cuNumeric automatically handles conversion between col-major and row-major for entry points which accept a Julia array. The only known API hole as a result is `Base.reshape` which we do not currently implement. 

**The Legate runtime builds a DAG asynchronously.** Calling `cuNumeric.zeros` or `A .+ B` records work into a task graph rather than blocking until every GPU kernel finishes. Results are materialized when you need them [for example `println`, `fetch`, or communicating with the Julia runtime (i.e., `Array(A)`)]. Hiding latency enables performant code.

For API details see [Initialization](https://julialegate.github.io/cuNumeric.jl/stable/api_initialization) and [NDArray Reference](https://julialegate.github.io/cuNumeric.jl/stable/api). For common performance pitfalls, see [Patterns to Avoid](https://julialegate.github.io/cuNumeric.jl/stable/perf/patterns_to_avoid).

### Kernel Fusion

Broadcast expressions fuse into a single kernel by default when on GPU. Prefer `@.` for multi-op elementwise code so every operator is dotted and the expression stays completely fused. Even just forgetting the `.` on unary negation (i.e., `y .= -a .+ b .* c`) will result in unfused code. Use the following pattern instead.

```julia
y .= @. -a + b * c
```

See [Kernel Fusion](https://julialegate.github.io/cuNumeric.jl/stable/perf/kernel_fusion) and [Debugging](https://julialegate.github.io/cuNumeric.jl/stable/debugging) for controls and diagnostics.

### The `@accelerate` macro

`@accelerate` fuses eligible GPU broadcasts within and across statements, then releases materialized temporary `NDArray`s after their last use on CPU or GPU. See [The `@accelerate` Macro](https://julialegate.github.io/cuNumeric.jl/stable/perf/accelerate/) for usage guidance.

### Benchmarks

Results and reproduction instructions live under [Benchmark Results](https://julialegate.github.io/cuNumeric.jl/stable/benchmarks/results) and [Running the Benchmarks](https://julialegate.github.io/cuNumeric.jl/stable/benchmarks/results#running-the-benchmarks). Run `./scripts/get_benchmarks.sh` to fetch the suite into `benchmark/`.

### Ecosystem Integrations

We implement a package extensions for [TensorOperations.jl](https://github.com/QuantumKitHub/TensorOperations.jl) and [Krylov.jl](https://github.com/juliasmoothoptimizers/krylov.jl). Simply load TensorOperations or Krylov alongside cuNumeric to take advantage! We also demonstrate composability with libraries like [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl). Check out the [examples](https://julialegate.github.io/cuNumeric.jl/stable/examples/ordinarydiffeq) in the documentation. Let us know if there's a feature in these libraries that doesn't work for you. 

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

println("Monte-Carlo Estimate: $(fetch(estimate))") # Should be ~ sqrt(pi)
```
More worked examples (initialization, Gray-Scott, …) are in the documentation sidebar under **Examples**.

### Known Limitations & Bugs

- There is no support for `Float16` or `ComplexF16` or `Complex{<:Integer}`
- `Base.reshape` is not supported
