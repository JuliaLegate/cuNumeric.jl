# Build Modes

`````@raw html
<!--
cuNumeric.jl gets its cuPyNumeric / Legate binaries from one of three providers, chosen through `CNPreferences` (writes `LocalPreferences.toml`, **restart Julia** after changing mode):

| Mode | When to use |
|:---|:---|
| **JLL (default)** | Normal installs; prebuilt artifacts from the Julia package server |
| **Developer** | Building or hacking the in-tree C++ wrapper, or a custom cuPyNumeric tree |
| **Conda** | Linking against an existing conda env that already has cuPyNumeric |
-->
`````

cuNumeric.jl uses prebuilt cuPyNumeric and Legate binaries. Choose the wrapper
build mode through `CNPreferences`. Restart Julia after changing it.

| Mode | When to use |
|:---|:---|
| **JLL (default)** | Normal installs using prebuilt artifacts from the Julia package server |
| **Developer** | Building the cuNumeric.jl C++ wrapper against the prebuilt dependencies |

Install `CNPreferences` on its own if you want to set the mode before adding `cuNumeric`:

```julia
using Pkg
Pkg.add("CNPreferences")
```

## Default build (JLLs)

```julia
using Pkg
Pkg.add("cuNumeric")
```

`````@raw html
<!--
If you previously used a custom build or conda build and would like to revert back to using prebuilt JLLs:
-->
`````

To switch back to the prebuilt wrapper:

```julia
using CNPreferences
CNPreferences.use_jll_binary()
```

Then restart Julia.

## Developer mode

Use developer mode to rebuild the cuNumeric.jl C++ wrapper after making changes. See the [Developer Mode](./developer_mode.md) docs for the full workflow.

Enable developer mode with:
```julia
using CNPreferences; CNPreferences.use_developer_mode()
```

After enabling developer mode or editing the wrapper, rebuild with
`Pkg.build("cuNumeric")` and restart Julia.

`````@raw html
<!-- ## Link Against Existing Conda Environment

> [!WARNING]
> This feature is not passing our CI currently. Please use with caution. We are failing to currently match proper versions of .so libraries together. Our hope is to get this functional for users already using Legate within conda.

Note, you need conda >= 24.1 to install the conda package. More installation details are found [here](https://docs.nvidia.com/cupynumeric/latest/installation.html).

````@eval
using Markdown
mm = Main.CUPYNUMERIC_MAJOR_MINOR
compat = Main.CUPYNUMERIC_JLL_COMPAT
Markdown.parse("""
Supported cuPyNumeric versions match the `cupynumeric_jll` major.minor in this repo's `Project.toml`. Currently that pin is `cupynumeric_jll = "$(compat)"`, so use the **$(mm)** conda line:

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

To update `LocalPreferences.toml` so that a local conda environment is used as the binary provider for cuPyNumeric run the following command. `conda_env` should be the absolute path to the conda environment (e.g., the value of CONDA_PREFIX when your environment is active). For example, this path is: `/home/JuliaLegate/.conda/envs/cupynumeric-gpu`.
```julia
using CNPreferences
using Pkg

CNPreferences.use_conda(ENV["CONDA_PREFIX"])  # absolute path, e.g. from CONDA_PREFIX
Pkg.build("cuNumeric")
```

Then **restart Julia** so the new mode loads. -->
`````
