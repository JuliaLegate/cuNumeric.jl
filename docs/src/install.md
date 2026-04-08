# Build Options

To make customization of the build options easier we have the `CNPreferences.jl` package to generate the `LocalPreferences.toml` which is read by the build script to determine which build option to use. CNPreferences.jl will also enforce that Julia is restarted for changes to take effect.


## Julia Installation

cuNumeric supports Julia 1.10 and 1.11. We recommend installing Julia with [juliaup](https://github.com/JuliaLang/juliaup):

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

## Default Build (jlls)

```julia
pkg> add cuNumeric
```
If you previously used a custom build or conda build and would like to revert back to using prebuilt JLLs, run the following command in the directory containing the Project.toml of your environment.

```julia
using CNPreferences; CNPreferences.use_jll_binary()
```

`CNPreferences` is a separate module so that it can be used to configure the build settings before `cuNumeric.jl` is added to your environment. To install it separately run

```julia
pkg> add CNPreferences
```

## Developer mode
> [!TIP]
> This gives the most flexibility in installs. It is meant for developing on cuNumeric.jl.

We support using a custom install version of cupynumeric. See https://docs.nvidia.com/cupynumeric/latest/installation.html for details about different install configurations, or building cupynumeric from source.

We require that you have a g++ capable compiler of C++ 20, and a recent version CMake >= 3.26.

To use developer mode,
```julia
using CNPreferences; CNPreferences.use_developer_mode(; use_cunumeric_jll=true, cunumeric_path=nothing)
```
By default `use_cunumeric_jll` will be set to true. However, you can set a custom branch and/or use a custom path of cupynumeric. By setting `use_cunumeric_jll=false`, you can set `cunumeric_path` to your custom install.
```julia
using CNPreferences; CNPreferences.use_developer_mode(;use_cunumeric_jll=false, cunumeric_path="/path/to/cupynumeric/root")

```

## Link Against Existing Conda Environment

> [!WARNING]
> This feature is not passing our CI currently. Please use with caution. We are failing to currently match proper versions of .so libraries together. Our hope is to get this functional for users already using Legate within conda.

Note, you need conda >= 24.1 to install the conda package. More installation details are found [here](https://docs.nvidia.com/cupynumeric/latest/installation.html).

```bash
# with a new environment
conda create -n myenv -c conda-forge -c cupynumeric
# into an existing environment
conda install -c conda-forge -c cupynumerice
```
Once you have the conda package installed, you can activate here.
```bash
conda activate [conda-env-with-cupynumeric]
```

To update `LocalPreferences.toml` so that a local conda environment is used as the binary provider for cupynumeric run the following command. `conda_env` should be the absolute path to the conda environment (e.g., the value of CONDA_PREFIX when your environment is active). For example, this path is: `/home/JuliaLegate/.conda/envs/cupynumeric-gpu`.
```julia
using CNPreferences; CNPreferences.use_conda("conda-env-with-legate");
Pkg.build()
```
