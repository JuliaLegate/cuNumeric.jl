# CNPreferences

Function reference for [`CNPreferences`](https://github.com/JuliaLegate/cuNumeric.jl/tree/main/lib/CNPreferences). For a usage guide that starts from the defaults, see [Configuration](./configuration.md).

All preference changes write `LocalPreferences.toml` and generally require a **fresh Julia process**.

## Build mode

```@docs
CNPreferences.use_jll_binary
CNPreferences.use_conda
CNPreferences.use_developer_mode
```

## Broadcast fusion

Defaults: fusion **on**, `FUSE_BROADCAST_MIN_OPS == 2`.

```@docs
CNPreferences.set_broadcast_fusion!
CNPreferences.enable_broadcast_fusion!
CNPreferences.disable_broadcast_fusion!
CNPreferences.set_broadcast_fusion_min_ops!
```

## Task scope names

Default: **off**.

```@docs
CNPreferences.set_task_scope_names!
CNPreferences.enable_task_scope_names!
CNPreferences.disable_task_scope_names!
```
