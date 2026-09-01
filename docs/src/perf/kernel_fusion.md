# Kernel Fusion

When CUDA is available cuNumeric.jl provides several methods to fuse operations into a single CUDA kernel. This can greatly improve performance and should be used whenever possible.

## Automatic Broadcast Fusion

Eligible broadcast expressions (i.e., `z .= cos.(x) .+ y`) are automatically fused into a single CUDA kernel.

We reccomend using Julia's `@.` macro to ensure the entire expression gets fused. A missing dot can result in poor performance!

```julia
# A missing dot on unary negation changes the expression and can prevent fusion.
y .= .-a .+ b .* c

# Prefer this form.
y .= @. -a + b * c
```

Automatic kernel fusion requires that:

```@raw html
<ol>
<li>Arrays have the same shape. Shape-mismatched broadcasts such as <code>matrix .+ vector</code> use the unfused path.</li>
<li>Expressions have at least two broadcast operations. Expressions like <code>y .= cos.(x)</code> with a single operation are unfused to reduce compilation overhead. This setting can be modified by calling <code>CNPreferences.set_broadcast_fusion_min_ops!(x::Int)</code> before launching cuNumeric. The default is <code>x = 2</code>.</li>
<li>A CUDA device is available.</li>
</ol>
```

Broadcast fusion can be disabled or re-enabled with CNPreferences as well:
```julia
CNPreferences.enable_broadcast_fusion!()           # default
CNPreferences.disable_broadcast_fusion!()
```

For more information on setting preferences see the CNPreferences [CNPreferences documentation](../api_preferences.md).

## Fuse Multiple Expressions with `@accelerate`

One function of the `@accelerate` macro is to analyze code and find temporary variables which can be elided via kernel fusion. In the example below `tmp` is unused outside of the `update!` function, so `@accelerate` will merge the two lines together.

```julia
@accelerate function update!(result, A, B, C)
    tmp = @. A + B
    result .= @. tmp * C + 1.0f0
    return result
end
```

The equivalent code is `result .= @. (A + B) * C + 1.0f0`. The rewrite is conservative: the producer must have one use, no intervening statement may invalidate its inputs, and the normal fusion requirements still apply.

This pattern requires the intermediate object to be temporary, and not pre-allocated. For example, if `tmp` was externally managed storage whose values were modified with `.=`, `@accelerate` would not be able to combine the expressions. For example, the following code would remain as two kernels.

```julia
tmp .= @. A + B
result .= @. tmp * C + 1.0f0
```

More details on `@accelerate` can be found in the [memory management](./reduce_allocations.md) docs.

## Other

To inspect the kernels emitted by broadcast fusion see these docs: [`BCAST_FUSION_DEBUG`](../debugging.md#inspect-fused-broadcasts-with-bcast_fusion_debug).
