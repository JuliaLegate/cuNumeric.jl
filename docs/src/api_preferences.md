# CNPreferences

Function reference for [`CNPreferences`](https://github.com/JuliaLegate/cuNumeric.jl/tree/main/lib/CNPreferences). Preferences write `LocalPreferences.toml` and generally require a **fresh Julia process**.

Out of the box (no `LocalPreferences.toml` changes):

| Setting | Default |
|---|---|
| Binary / build mode | **JLL** prebuilt binaries |
| Broadcast fusion | **on** |
| `FUSE_BROADCAST_MIN_OPS` | **2** (single native operations stay unfused) |
| Task scope names | **off** |
| `MIN_SOLVE_MATRIX_SIZE` | **2048** rows |
| `MIN_SOLVE_TILE_SIZE` | **512** |
| `MIN_CHOLESKY_MATRIX_SIZE` | **8192** rows |
| `MIN_CHOLESKY_TILE_SIZE` | **2048** |
| `MIN_QR_MATRIX_SIZE` | **1048576** elements |
| `QR_TILE_SIZE` | **128** |
| `MAX_CHOLESKY_TILES_PER_PROC` | **4** |

Build-mode setup (JLL / conda / developer) is documented under [Build Modes](./install.md).

## Linear algebra

`set_linalg!` accepts the constant names above as keywords. All values must be
positive integers; unspecified settings are unchanged. Restart Julia to load
the new constants. `cuNumeric.versioninfo()` shows the effective values.

```julia
CNPreferences.set_linalg!(; MIN_SOLVE_MATRIX_SIZE=4096, MIN_SOLVE_TILE_SIZE=512)
```

See [Distributed solves and factorizations](./linalg.md#distributed-solves-and-factorizations)
for selection rules and tuning details.

```@docs
CNPreferences.set_linalg!
```

## Build mode

```@docs
CNPreferences.use_jll_binary
CNPreferences.use_conda
CNPreferences.use_developer_mode
```

## Broadcast fusion

Defaults: fusion **on**, `FUSE_BROADCAST_MIN_OPS == 2`.

```julia
using CNPreferences

CNPreferences.enable_broadcast_fusion!()           # default
CNPreferences.disable_broadcast_fusion!()
CNPreferences.set_broadcast_fusion_min_ops!(2)     # default
CNPreferences.set_broadcast_fusion_min_ops!(1)     # also fuse single-ops
```

`set_broadcast_fusion_min_ops!` counts `Broadcasted` nodes (ops) in the tree:

- **`2` (default):** fuse multi-op trees such as `y .= @. a * b + c`. Single native operations like `y .= cos.(x)` use the C-API path. Functions without a native implementation are fused on the GPU when their broadcast arguments are eligible.
- **`1`:** fuse every eligible expression, including single-ops.

Set the preference in one Julia process, then start a fresh process to use it.

```@docs
CNPreferences.set_broadcast_fusion!
CNPreferences.enable_broadcast_fusion!
CNPreferences.disable_broadcast_fusion!
CNPreferences.set_broadcast_fusion_min_ops!
```

## Task scope names

Default: **off**. Optional Legate task-scope naming for debugging. When on, cuNumeric wraps many ops in `Legate.with_scope` so provenance strings (for example `matmul`, `zeros`, or fused `broadcast.<expr>`) appear in Legate logs and profiles. Pair this with `--logging legate=debug --log-to-file` (or `--profile`) in `LEGATE_CONFIG`; see [Debugging](./debugging.md#trace-legate-runtime-work).

```julia
using CNPreferences
CNPreferences.enable_task_scope_names!()
CNPreferences.disable_task_scope_names!()  # default
```

Restart Julia after changing this preference (it is compile-time in cuNumeric.jl).

```@docs
CNPreferences.set_task_scope_names!
CNPreferences.enable_task_scope_names!
CNPreferences.disable_task_scope_names!
```
