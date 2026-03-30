# cuNumeric.jl

[![Documentation dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://julialegate.github.io/cuNumeric.jl/dev/)
[![codecov](https://codecov.io/github/julialegate/cuNumeric.jl/branch/main/graph/badge.svg)](https://app.codecov.io/github/JuliaLegate/cuNumeric.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)


The cuNumeric.jl package wraps the [cuPyNumeric](https://github.com/nv-legate/cupynumeric) C++ API from NVIDIA to bring simple distributed computing on GPUs and CPUs to Julia! We provide a simple array abstraction, the `NDArray`, which supports most of the operations you would expect from a normal Julia array.

> [!WARNING]  
> cuNumeric.jl is under active development. This is an alpha API and is subject to change. Stability is not guaranteed until the first official release. We are actively working to improve the build experience to be more seamless and Julia-friendly.

### Quick Start
cuNumeric.jl can be installed with the Julia package manager. From the Julia REPL, type `]` to enter the Pkg REPL mode and run:
```julia
pkg> add cuNumeric
```
Or, using the `Pkg` API:
```julia
using Pkg; Pkg.add(url = "https://github.com/JuliaLegate/cuNumeric.jl", rev = "main")
```
The first run might take awhile as it has to install multiple large dependencies such as the CUDA SDK (if you have an NVIDIA GPU). For more install instructions, please visit out install guide in the documentation.

To see information about your cuNumeric install run the `versioninfo` function.

```julia
cuNumeric.versioninfo()
```

### Monte-Carlo Example
```julia
using cuNumeric

integrand = (x) -> exp.(-x.^2)

N = 1_000_000

x_max = 10.0f0
domain = [-x_max, x_max]
Ω = domain[2] - domain[1]

samples = Ω*cuNumeric.rand(N) .- x_max 
estimate = (Ω/N) * sum(integrand(samples))

println("Monte-Carlo Estimate: $(estimate)")
```

### Requirements

We require an x86 Linux platform and Julia 1.10 or 1.11. For GPU support we require an NVIDIA GPU and a CUDA driver which supports CUDA 13.0. ARM support is theoretically possible, but we do not make binaries or test on ARM. Please open an issue if ARM support is of interest.
