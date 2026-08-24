# Kernel Fusion

When CUDA is available, cuNumeric can compile an eligible nested broadcast tree into one PTX kernel instead of launching one operation at a time. CPU execution follows the normal unfused path.

Prefer Julia's `@.` macro for multi-operation elementwise expressions so every operator is dotted:

```julia
# A missing dot on unary negation changes the expression and can prevent fusion.
y .= .-a .+ b .* c

# Prefer this form.
y .= @. -a + b * c
```

Fusion requires array leaves with the same shape and at least `FUSE_BROADCAST_MIN_OPS` broadcast operations (default: 2). Shape-mismatched broadcasts such as `matrix .+ vector` use the unfused path.

## Fuse across statements with `@accelerate`

An ordinary assignment lets `@accelerate` substitute a single-use producer into its consumer:

```julia
@accelerate function update!(result, A, B, C)
    tmp = @. A + B
    result .= @. tmp * C + 1.0f0
    return result
end
```

This can become the equivalent of `result .= @. (A + B) * C + 1.0f0`. The rewrite is conservative: the producer must have one use, no intervening statement may invalidate its inputs, and the normal fusion requirements still apply.

Do not preallocate a single-use intermediate with `.=` merely to avoid allocation:

```julia
tmp .= @. A + B
result .= @. tmp * C + 1.0f0
```

The mutation of `tmp` is observable, so it remains a kernel boundary. Keep preallocation when the intermediate is reused or its mutation must be visible.

The `begin` form has different ownership semantics: named bindings remain live. On CUDA, an eligible same-shape chain can still be emitted as one multi-output kernel that materializes each binding. See [Accelerate Array Code](./reduce_allocations.md) for all macro forms.

## Configure and inspect fusion

Set preferences in one Julia process, then restart Julia:

```julia
using CNPreferences

CNPreferences.enable_broadcast_fusion!()          # default
CNPreferences.disable_broadcast_fusion!()
CNPreferences.set_broadcast_fusion_min_ops!(2)    # default
CNPreferences.set_broadcast_fusion_min_ops!(1)    # include single-op broadcasts
```

The default threshold avoids PTX compilation overhead when there is little work to combine. See [CNPreferences](../api_preferences.md) for preference details and [`BCAST_FUSION_DEBUG`](../debugging.md#inspect-fused-broadcasts-with-bcast_fusion_debug) to confirm which path ran.
