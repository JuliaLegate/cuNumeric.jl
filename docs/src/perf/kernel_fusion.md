# Kernel Fusion

On CUDA, nested broadcast expressions are fused into a single PTX kernel when fusion is enabled (the default). There is no separate `@fuse` macro. You write ordinary Julia broadcast code, and cuNumeric compiles eligible trees into one kernel instead of launching one op at a time.

Prefer Julia's `@.` macro for multi-op elementwise expressions. Placing `.` on every operator by hand is easy to get wrong: missing a dot on unary negation or addition silently changes the meaning, and it can also break fusion by splitting work into the wrong ops.

```julia
# Easy to miss the dot on negation
y .= .-a .+ b .* c

# Prefer: @. dots every operator, which is clearer and fusion-friendly
y .= @. -a + b * c
```

Fusion applies when CUDA is available, the array leaves share the same shape, and the expression has at least `FUSE_BROADCAST_MIN_OPS` ops (default 2). Otherwise cuNumeric falls back to evaluating one op at a time. Shape-mismatched broadcasts such as `matrix .+ vector` use the unfused path.

Toggle fusion through `CNPreferences` (restart Julia after changing these):

```julia
using CNPreferences

CNPreferences.enable_broadcast_fusion!()          # default
CNPreferences.disable_broadcast_fusion!()
CNPreferences.set_broadcast_fusion_min_ops!(2)    # default
CNPreferences.set_broadcast_fusion_min_ops!(1)    # also fuse single-ops
```

What `set_broadcast_fusion_min_ops!` controls:

- **`2` (default):** only trees with two or more ops fuse. Example: `y .= @. a * b + c` can fuse; `y .= cos.(x)` does not. Keeping single-ops on the unfused C-API path avoids PTX compile overhead when there is little to gain.
- **`1`:** every eligible broadcast can fuse, including unary / single-op forms. Prefer this when you want uniform fused behavior (for example in tests) rather than for typical apps.

The threshold counts `Broadcasted` nodes in the expression tree. Changing it requires a Julia restart. `CUNUMERIC_FUSE_BROADCAST_MIN_OPS` can override the preference in CI; see [CNPreferences](../api_preferences.md).

To inspect a fused launch or a lifetime rewrite, see [Debugging](../debugging.md). For the implementation pipeline, see [Internals](../internals.md).
