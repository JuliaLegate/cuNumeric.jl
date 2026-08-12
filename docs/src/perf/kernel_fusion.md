# Kernel Fusion

On CUDA, nested broadcast expressions are fused into a single PTX kernel when fusion is enabled (the default). There is no separate `@fuse` macro. You write ordinary Julia broadcast code, and cuNumeric compiles eligible trees into one kernel instead of launching one op at a time.

Prefer Julia's `@.` macro for multi-op elementwise expressions. Placing `.` on every operator by hand is easy to get wrong: missing a dot on unary negation or addition silently changes the meaning, and it can also break fusion by splitting work into the wrong ops.

```julia
# Easy to miss the dot on negation
y .= .-a .+ b .* c

# Prefer: @. dots every operator, which is clearer and fusion-friendly
y .= @. -a + b * c
```

## Avoid preallocated intermediate broadcast buffers

Preallocation is useful for a final output or a buffer that must persist across
iterations. It can be counterproductive for a single-use intermediate inside
`@analyze_lifetimes`, however. An in-place `.=` assignment is an observable
mutation, so inter-statement broadcast fusion treats it as a kernel boundary:

```julia
tmp = cuNumeric.zeros(Float32, N, N)
result = cuNumeric.zeros(Float32, N, N)

@analyze_lifetimes begin
    tmp .= @. A + B
    result .= @. tmp * C + 1.0f0
end
```

This materializes `tmp` before the second expression and requires separate
kernel launches. Instead, use an ordinary assignment for a single-use
intermediate and keep `.=` for the final destination:

```julia
result = cuNumeric.zeros(Float32, N, N)

@analyze_lifetimes begin
    tmp = @. A + B
    result .= @. tmp * C + 1.0f0
end
```

The inter-statement pass can substitute `tmp` into its only consumer, producing
the equivalent of `result .= @. (A + B) * C + 1.0f0`. The intermediate is never
materialized, so the full expression can run as one fused kernel.

This rewrite is intentionally conservative: the intermediate must have one
use, no intervening statement may invalidate its inputs, and all normal fusion
requirements still apply. Keep preallocation when an intermediate is reused,
must preserve mutation semantics, or cannot be fused. Use
[`BCAST_FUSION_DEBUG`](../debugging.md#inspect-fused-broadcasts-with-bcast_fusion_debug)
to confirm whether the rewrite occurred.

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

The threshold counts `Broadcasted` nodes in the expression tree. Set it through `CNPreferences`, then restart Julia. See [CNPreferences](../api_preferences.md).

To inspect a fused launch or a lifetime rewrite, see [Debugging](../debugging.md). For the implementation pipeline, see [Internals](../internals.md).
