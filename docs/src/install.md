# Build Options

To make customization of the build options easier we have the `CNPreferences.jl` package to generate the `LocalPreferences.toml` which is read by the build script to determine which build option to use. CNPreferences.jl will also enforce that Julia is restarted for changes to take effect.

## Default Build (jlls)

By default cuNumeric.jl will use [Binary Builder](https://github.com/JuliaPackaging/Yggdrasil) to install cuNumeric.jl

```
Pkg.add(url = "https://github.com/JuliaLegate/cuNumeric.jl")
```

If you previously used a custom build or conda build and would like to revert back to using prebuilt JLLs, run the following command in the directory containing the Project.toml of your environment.


```julia
julia --project -e 'using CNPreferences; CNPreferences.use_jll_binary()'
```

`CNPreferences` is a separate module so that it can be used to configure the build settings before `cuNumeric.jl` is added to your environment. To install it separately run

```julia
using Pkg; Pkg.add(url = "https://github.com/JuliaLegate/cuNumeric.jl", subdir="lib/CNPreferences")
```

By default, this will also revert any LegatePreferences you have set. It will revert Legate.jl to use JLLs. You can disable this behavior with `transitive = false` in the `use_jll_binary()` function.

## Developer mode

[!WARNING] 
This gives the most flexibility in installs. It is meant for developing on cuNumeric.jl. By default, this does not set any LegatePreferences. 

We require that you have the cuda driver `libcuda.so` on your path, cuda runtime `libcudart.so`,  g++ capable compiler of C++ 20, and a recent version CMake >= 3.26.

To use developer mode, 
```julia
julia --project -e 'using CNPreferences; CNPreferences.use_developer_mode(; wrapper_branch="main", use_cupynumeric_jll=true, cupynumeric_path=nothing)'
```
This will clone [cunumeric_jl_wrapper](https://github.com/JuliaLegate/cunumeric_jl_wrapper) into cuNumeric.jl/deps and build from src. By default `use_cupynumeric_jll` will be set to true and `wrapper_branch` will be set to "main". However, you can set a custom branch and/or use a custom path of cupynumeric. By using disabling `use_cupynumeric_jll`, you can set `cupynumeric_path` to your custom root. 

Developer mode by default is not transitive to LegatePreferences. This means setting cuNumeric.jl to devloper mode has no impact on Legate.jl. To have both libraries on developer mode, you need to set Legate preferences manually. 

```julia
julia --project -e 'using LegatePreferences; LegatePreferences.use_developer_mode(; wrapper_branch="main", use_legate_jll=true, legate_path=nothing)'
```
LegatePreferences has similar kwargs and behavior as CNPreferences. This will clone [legate_jl_wrapper](https://github.com/JuliaLegate/legate_jl_wrapper). By default `use_legate_jll` is set to true and `wrapper branch` is set to "main" You can disable the jll and set a custom legate root with `legate_path`. 

## Link Against Existing Conda Environment

[!WARNING] 
This feature is not passing our CI currently. Please use with caution. We are failing to currently match proper versions of .so libraries together. Our hope is to get this functional for users already using cuPyNumeric within conda. 

To update `LocalPreferences.toml` so that a local conda environment is used as the binary provider for cupynumeric run the following command. `conda_env` should be the absolute path to the conda environment (e.g., the value of CONDA_PREFIX when your environment is active). For example, this path is: `/home/JuliaLegate/.conda/envs/cunumeric-gpu`.

```julia
julia --project -e 'using CNPreferences; CNPreferences.use_conda("<env-path>")'
```
A common reason to link against an existing environment is to avoid cupynumeric re-installing the entire CUDA toolkit. To make your own conda environment built off an existing CUDA install run the following with whatever version of CUDA is on the `LD_LIBRARY_PATH` in place of 12.2.

```bash 
conda create --name myenv 
conda activate myenv
CONDA_OVERRIDE_CUDA="12.4" \
  conda install -c conda-forge -c legate cupynumeric
```

By default, this will also revert any LegatePreferences you have set. It will revert Legate.jl to use JLLs. You can disable this behavior with `transitive = false` in the `use_conda()` function.
