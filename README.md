# cuNumeric.jl

[![Documentation dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ejmeitz.github.io/cuNumeric.jl/dev)
[![Build status](https://ci.appveyor.com/api/projects/status/973jtue9itgvvlc7?svg=true)](https://ci.appveyor.com/project/ejmeitz/cunumeric-jl)
[![codecov](https://codecov.io/github/ejmeitz/cuNumeric.jl/branch/main/graph/badge.svg)](https://app.codecov.io/github/ejmeitz/cuNumeric.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

The cuNumeric.jl package wraps the [cuPyNumeric](https://github.com/nv-legate/cupynumeric) C++ API from NVIDIA to bring simple distributed computing on GPUs and CPUs to Julia! We provide a simple array abstraction, the `NDArray`, which supports most of the operations you would expect from a normal Julia array.

This project is in alpha and we do not commit to anything necessarily working as you would expect. The current build process requires several external dependencies which are not registered on BinaryBuilder.jl yet. The build instructions and minimum pre-requesites are as follows:

## Minimum prereqs
- g++ capable of C++20
- CUDA 12.2
- Python 3.10
- Ubuntu 20.04 or RHEL 8
- Julia 1.10
- CMake 3.26.4 

### 1. Install Julia through [JuliaUp](https://github.com/JuliaLang/juliaup)
```
curl -fsSL https://install.julialang.org | sh -s -- --default-channel 1.10
```

This will install version 1.10 by default since that is what we have tested against. To verify 1.10 is the default run either of the following (you may need to source bashrc):
```bash
juliaup status
julia --version
```

If 1.10 is not your default, please set it to be the default. Other versions of Julia are untested.
```bash
juliaup default 1.10
```

### 2. Download cuNumeric.jl
cuNumeric.jl is not on the general registry yet. To add cuNumeric.jl to your environment run:
```julia
using Pkg; Pkg.add(url = "https://github.com/JuliaLegate/cuNumeric.jl", rev = "main")
```

The `rev` option can be main or any tagged version.  By default, this will use [legate_jll](https://github.com/ejmeitz/legate_jll.jl) and build cuPyNumeric from source. In [2b](#2b-use-preinstalled-version-of-cupynumeric) and [2c](#2c-use-a-conda-environment-to-install-cunumericjl), we show different installation methods. Ensure that the enviroment variables are correctly set for custom builds.


To contribute to cuNumeric.jl, we recommend cloning the repository and manually triggering the build process with `Pkg.build` or adding it to one of your existing environments with `Pkg.develop`.
```bash
git clone https://github.com/JuliaLegate/cuNumeric.jl.git
cd cuNumeric.jl
julia -e 'using Pkg; Pkg.activate(".") Pkg.resolve(); Pkg.build()'
```

#### 2a. Known build issue (unregistered packages)
Legate.jl and Legate_jll are both unregistered. You may need to "hack"  if you run into build issues. We hope this issue goes away once we have these packages registered.
```julia
] activate .
] rm Legate
] add https://github.com/ejmeitz/legate_jll.jl/
] add https://github.com/JuliaLegate/Legate.jl # will fail to precompile
] build # builds Legate.jl and cuNumeric.jl
```
#### 2b. Use preinstalled version of [cuPyNumeric](https://github.com/nv-legate/cupynumeric)
We support using a custom install version of cuPyNumeric. See https://docs.nvidia.com/cupynumeric/latest/installation.html for details about different install configurations, or building cuPyNumeric from source.
```bash
export CUNUMERIC_CUSTOM_INSTALL=1
export CUNUMERIC_CUSTOM_INSTALL_LOCATION="/home/user/path/to/cupynumeric-install-dir"
```
```julia
using Pkg; Pkg.add(url = "https://github.com/JuliaLegate/cuNumeric.jl", rev = "main")
```
cuNumeric.jl depends on [Legate.jl](https://github.com/JuliaLegate/Legate.jl). To use a custom Legate install, follow the instructions [here](https://github.com/JuliaLegate/Legate.jl?tab=readme-ov-file#2b-use-preinstalled-version-of-legate). 

#### 2c. Use a conda environment to install cuNumeric.jl
Note, you need conda >= 24.1 to install the conda package. More installation details are found [here](https://docs.nvidia.com/cupynumeric/latest/installation.html).
```bash
# with a new environment
conda create -n myenv -c conda-forge -c legate cupynumeric
# into an existing environment
conda install -c conda-forge -c legate cupynumeric
```
Once you have the conda package installed, you can activate here. 
```bash
conda activate [conda-env-with-cupynumeric]
export CUNUMERIC_LEGATE_CONDA_INSTALL=1
```
```julia
using Pkg; Pkg.add(url = "https://github.com/JuliaLegate/cuNumeric.jl", rev = "main")
```
### 3. Test the Julia Package
Run this command in the Julia environment where cuNumeric.jl is installed.
```julia
using Pkg; Pkg.test("cuNumeric")
```

With everything working, its the perfect time to checkout some of our [examples](https://ejmeitz.github.io/cuNumeric.jl/dev/examples/)!



## TO-DO List of Missing Important Features
- Implement `unary_reduction` over arbitrary dims
- Out-parameter `binary_op`
- Replace `as_type` with `Base.convert`
- Integer powers (e.g x^3)
- Support Ints on methods that takes floats
- Programatic manipulation of Legate hardware config (not currently possible)
- Float32 random number generation (not possible in current C++ API)
- Normal random numbers (not possible in current C++ API)
- Add Aqua.jl to CI to ensure we didn't pirate any types
- Fix CodeCov reports
- Fix cuNumeric.jl error in CI (requires unreleased CuPyNumeric)

## Contact
For technical questions, please either contact 
`krasow(at)u.northwestern.edu` OR
`emeitz(at)andrew.cmu.edu`

If the issue is building the package, please include the `build.log` and `.err` files found in `cuNumeric.jl/pkg/deps/` 

