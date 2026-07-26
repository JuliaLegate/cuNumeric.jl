# Build Modes

This page covers installing Julia and choosing a binary provider for cuNumeric. For defaults and a short tour of `CNPreferences`, start with [Configuration](./configuration.md).

`CNPreferences` writes `LocalPreferences.toml` and requires a Julia restart when the build mode changes.

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
> For rebuilding `lib/cunumeric_jl_wrapper` after C++ changes, see [Developer Mode](./developer_mode.md).

We support using a custom install version of cupynumeric. See https://docs.nvidia.com/cupynumeric/latest/installation.html for details about different install configurations, or building cupynumeric from source.

We require that you have a g++ capable compiler of C++ 20, and a recent version CMake >= 3.26.

To use developer mode,
```julia
using CNPreferences; CNPreferences.use_developer_mode(; use_jll=true, path=nothing)
```
By default `use_jll` will be set to true. However, you can use a custom path of cupynumeric. By setting `use_jll=false`, you can set `path` to your custom install.
```julia
using CNPreferences; CNPreferences.use_developer_mode(;use_jll=false, path="/path/to/cupynumeric/root")

```

After enabling developer mode (and after any wrapper edits), rebuild with `Pkg.build("cuNumeric")` and restart Julia. Details are on [Developer Mode](./developer_mode.md).

## Link Against Existing Conda Environment

> [!WARNING]
> This feature is not passing our CI currently. Please use with caution. We are failing to currently match proper versions of .so libraries together. Our hope is to get this functional for users already using Legate within conda.

Note, you need conda >= 24.1 to install the conda package. More installation details are found [here](https://docs.nvidia.com/cupynumeric/latest/installation.html).

````@eval
using Markdown
mm = Main.CUPYNUMERIC_MAJOR_MINOR
compat = Main.CUPYNUMERIC_JLL_COMPAT
Markdown.parse("""
Supported cupynumeric versions match the `cupynumeric_jll` major.minor in this repo's `Project.toml`. Currently that pin is `cupynumeric_jll = "$(compat)"`, so use the **$(mm)** conda line:

```bash
# with a new environment
conda create -n myenv -c conda-forge -c cupynumeric cupynumeric=$(mm)
# into an existing environment
conda install -c conda-forge -c cupynumeric cupynumeric=$(mm)
```
""")
````
Once you have the conda package installed, you can activate here.
```bash
conda activate [conda-env-with-cupynumeric]
```

To update `LocalPreferences.toml` so that a local conda environment is used as the binary provider for cupynumeric run the following command. `conda_env` should be the absolute path to the conda environment (e.g., the value of CONDA_PREFIX when your environment is active). For example, this path is: `/home/JuliaLegate/.conda/envs/cupynumeric-gpu`.
```julia
using CNPreferences
CNPreferences.use_conda(ENV["CONDA_PREFIX"])  # absolute path, e.g. from CONDA_PREFIX
Pkg.build()
```
