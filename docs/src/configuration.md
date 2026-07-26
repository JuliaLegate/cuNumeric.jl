# Configuration

cuNumeric is configured through the `CNPreferences` package. Preferences are written to `LocalPreferences.toml` and take effect after you **restart Julia** (many values are compile-time constants).

Install `CNPreferences` on its own if you want to configure builds before adding `cuNumeric`:

```julia
pkg> add CNPreferences
```

## Defaults

Out of the box, with no `LocalPreferences.toml` changes:

| Setting | Default |
|---|---|
| Binary / build mode | **JLL** prebuilt binaries (`use_jll_binary`) |
| Broadcast fusion | **on** |
| `FUSE_BROADCAST_MIN_OPS` | **2** (single-op broadcasts stay unfused) |
| Task scope names | **off** |

Typical first install:

```julia
pkg> add cuNumeric
```

That uses JLLs. Broadcast fusion is already enabled. You do not need to call any `CNPreferences` setters unless you want a different build or you want to change fusion / debug prefs.

To return to the default JLL build after trying conda or developer mode:

```julia
using CNPreferences
CNPreferences.use_jll_binary()
```

Then restart Julia. You may also need `Pkg.build()` if you left a non-JLL mode.

## What you can configure

### Build mode

Choose where cupynumeric / Legate binaries come from:

- **JLL (default):** prebuilt artifacts from the Julia package server
- **Conda:** link against an existing conda environment that already has cupynumeric
- **Developer:** build wrappers from source, optionally against a custom cupynumeric tree

Details and Julia version notes are in [Build Modes](./install.md).

### Broadcast fusion

Control whether nested `@.` / dotted broadcasts fuse into one CUDA kernel:

```julia
using CNPreferences

CNPreferences.enable_broadcast_fusion!()           # default
CNPreferences.disable_broadcast_fusion!()
CNPreferences.set_broadcast_fusion_min_ops!(2)     # default
CNPreferences.set_broadcast_fusion_min_ops!(1)     # also fuse single-ops
```

`set_broadcast_fusion_min_ops!` sets how large a broadcast tree must be before fusion runs:

- **`2` (default):** fuse expressions with at least two ops, such as `y .= @. a * b + c`. Trivial single-op forms like `y .= cos.(x)` or `y .= @. -a` stay on the mature unfused C-API path. That avoids paying PTX compile cost when fusion would not save launches.
- **`1`:** fuse every eligible expression, including single-ops. Useful when you want one code path for testing, or when even unary kernels should go through the fused PTX launcher.

Count is the number of `Broadcasted` nodes (ops) in the tree, not the number of arrays. Restart Julia after changing these. See [Kernel Fusion](./perf/kernel_fusion.md) and [Debugging](./debugging.md).

### Task scope names

Optional Legate task-scope naming for debugging (default off):

```julia
using CNPreferences
CNPreferences.enable_task_scope_names!()
CNPreferences.disable_task_scope_names!()  # default
```

## Using a conda build instead of JLLs

> [!WARNING]
> Conda linking is not currently passing CI. Use with caution. Shared-library version matching is still rough.

````@eval
using Markdown
mm = Main.CUPYNUMERIC_MAJOR_MINOR
compat = Main.CUPYNUMERIC_JLL_COMPAT
Markdown.parse("""
Install a cupynumeric build that matches the `cupynumeric_jll` major.minor pinned in this package's `Project.toml`. Currently that is **$(mm)** (`cupynumeric_jll = "$(compat)"`), so use conda packages from the **$(mm)** line (for example `cupynumeric=$(mm)`).
""")
````

1. Create or update a conda env (conda `>= 24.1` recommended):

````@eval
using Markdown
mm = Main.CUPYNUMERIC_MAJOR_MINOR
Markdown.parse("""
```bash
conda create -n myenv -c conda-forge -c cupynumeric cupynumeric=$(mm)
conda activate myenv
```
""")
````

2. Point `CNPreferences` at the **absolute** path of that env (the value of `CONDA_PREFIX` while the env is active), then rebuild:

```julia
using CNPreferences
using Pkg

CNPreferences.use_conda(ENV["CONDA_PREFIX"])  # e.g. "/home/you/.conda/envs/myenv"
# or: CNPreferences.use_conda("/absolute/path/to/conda/env")

Pkg.build("cuNumeric")
```

3. **Restart Julia** so the new mode loads.

To go back to JLLs:

```julia
using CNPreferences
CNPreferences.use_jll_binary()
# restart Julia, then Pkg.build() if needed
```

## Related pages

- [Hardware](./configuration/hardware.md): Legate `LEGATE_*` resource ENV vars
- [Build Modes](./install.md): Julia install, JLL, developer mode, and conda in more detail
- [CNPreferences](./api_preferences.md): function reference
- [Kernel Fusion](./perf/kernel_fusion.md): `@.` usage, fusion thresholds, and related tips
