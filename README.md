# cuNumeric.jl

[![Documentation dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://julialegate.github.io/cuNumeric.jl/dev/)
[![codecov](https://codecov.io/github/julialegate/cuNumeric.jl/branch/main/graph/badge.svg)](https://app.codecov.io/github/JuliaLegate/cuNumeric.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> [!WARNING]  
> Leagte.jl and cuNumeric.jl are under active development at the moment. This is a pre-release API and is subject to change. Stability is not guaranteed until the first official release. We are actively working to improve the build experience to be more seamless and Julia-friendly. In parallel, we're developing a comprehensive testing framework to ensure reliability and robustness. Our public beta launch is targeted for Fall 2025.

The cuNumeric.jl package wraps the [cuPyNumeric](https://github.com/nv-legate/cupynumeric) C++ API from NVIDIA to bring simple distributed computing on GPUs and CPUs to Julia! We provide a simple array abstraction, the `NDArray`, which supports most of the operations you would expect from a normal Julia array.

This project is in alpha and we do not commit to anything necessarily working as you would expect. The current build process requires several external dependencies which are not registered on BinaryBuilder.jl yet. The build instructions and minimum pre-requesites are as follows:

### Minimum prereqs
- Ubuntu 20.04 or RHEL 8
- Julia 1.11

### 1. Install Julia through [JuliaUp](https://github.com/JuliaLang/juliaup)
```
curl -fsSL https://install.julialang.org | sh -s -- --default-channel 1.11
```

This will install version 1.11 by default since that is what we have tested against. To verify 1.11 is the default run either of the following (you may need to source bashrc):
```bash
juliaup status
julia --version
```

If 1.11 is not your default, please set it to be the default. Other versions of Julia are untested.
```bash
juliaup default 1.11
```

### 2. Download cuNumeric.jl (quick setup)
cuNumeric.jl is not on the general registry yet. To add cuNumeric.jl to your environment run:
```julia
using Pkg; Pkg.develop(url = "https://github.com/JuliaLegate/cuNumeric.jl")
```
By default, this will use [legate_jll](https://github.com/JuliaBinaryWrappers/legate_jll.jl/) and [cupynumeric_jll](https://github.com/JuliaBinaryWrappers/cupynumeric_jll.jl/). 

For more build configurations and options, please visit our [installation guide](https://julialegate.github.io/cuNumeric.jl/dev/install).

#### 2b. Contributing to cuNumeric.jl
To contribute to cuNumeric.jl, we recommend cloning the repository and adding it to one of your existing environments with `Pkg.develop`.
```bash
git clone https://github.com/JuliaLegate/cuNumeric.jl.git 
julia --project=. -e 'using Pkg; Pkg.develop(path = "cuNumeric.jl/lib/CNPreferences")'
julia --project=. -e 'using Pkg; Pkg.develop(path = "cuNumeric.jl")'
julia --project=. -e 'using CNPreferences; CNPreferences.use_developer_mode()'
julia --project=. -e 'using Pkg; Pkg.build()'
```

To learn more about contributing to Legate.jl, check out the [Legate.jl README.md](https://github.com/JuliaLegate/Legate.jl?tab=readme-ov-file#2-download-legatejl)

### 3. Test the Julia Package
Run this command in the Julia environment where cuNumeric.jl is installed.
```julia
using Pkg; Pkg.test("cuNumeric")
```
With everything working, its the perfect time to checkout some of our [examples](https://julialegate.github.io/cuNumeric.jl/dev/examples)!


## Contact
For technical questions, please either contact 
`krasow(at)u.northwestern.edu` OR
`emeitz(at)andrew.cmu.edu`

If the issue is building the package, please include the `build.log` and `.err` files found in `cuNumeric.jl/deps/` 

