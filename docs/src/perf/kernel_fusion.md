# Kernel Fusion

On a GPU, cuNumeric compiles an eligible broadcast expression such as
`z .= cos.(x) .+ y` into a single CUDA kernel instead of one launch per
operation. Without a GPU, every operation runs separately.

Use `@.` so the whole expression is dotted. A missing dot splits the
expression:

```julia
y .= .-a .+ b .* c     # easy to miss a dot
y .= @. -a + b * c     # prefer this
```

## When an expression fuses

- **Same shape.** Arrays of different shapes broadcast by Julia's rules but
  run unfused. Future versions may fuse them.
- **At least two operations.** `y .= cos.(x)` uses cuPyNumeric's own kernel.
  A single call to your own function, such as `f.(x)`, still fuses.
- **Plain scalars** (`2.0f0`, `α`) are passed to the kernel at run time, so
  changing them doesn't recompile.

## Preferences

```julia
using CNPreferences
CNPreferences.disable_broadcast_fusion!()        # enable_broadcast_fusion!() is the default
CNPreferences.set_broadcast_fusion_min_ops!(1)   # fuse single operations too (default 2)
```

Restart Julia after changing a preference; see [CNPreferences](../api_preferences.md).

## Fusing across statements

Fusion works within one expression. To fuse a chain of statements, and free
temporaries early, use [`@accelerate`](./accelerate.md).

To see the kernels that fusion generates, set
[`BCAST_FUSION_DEBUG`](../debugging.md#inspect-fused-broadcasts-with-bcast_fusion_debug).
