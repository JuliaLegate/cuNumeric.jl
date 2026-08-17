# The `@accelerate` Macro

`@accelerate` optimizes *straight-line* array code: a fixed sequence of statements
with no branches, loops, jumps, `try`, or nested functions. Ordinary calls are
opaque boundaries; general control flow is not rewritten.

Within that restricted body, it performs:

<ol>
<li><strong>Fusion within a broadcast expression.</strong> On CUDA, an eligible dotted expression such as <code>@. A + B * C</code> can run as one kernel. CPU execution uses the normal unfused path.</li>
<li><strong>Fusion across broadcast statements.</strong> A single-use broadcast result can be substituted into its consumer, producing fewer GPU kernel launches.</li>
<li><strong>Temporary lifetime analysis.</strong> After rewriting the code, the macro releases materialized, non-returned <code>NDArray</code>s after their final use on CPU or GPU.</li>
</ol>

These jobs must happen together: an intermediate that fuses into its consumer is never allocated, while an intermediate that cannot fuse is materialized and then released after its last use.

## Use the function form by default

Annotate a reusable straight-line function:

```julia
@accelerate function update!(C, A, B)
    combined = @. A + B
    C .= @. 2.0f0 * combined
    return C
end
```

On an eligible GPU path, `combined` can be folded into the second broadcast so the chain runs as one kernel. On CPU, or when fusion is ineligible, `combined` is materialized and released after the update. Function arguments belong to the caller and are never released by `@accelerate`; returned values also remain valid.

## Choose a form

The forms differ in which values must remain available, which determines how aggressively the macro may fuse or release intermediates.

| Form | Use it when | Fusion and lifetime behavior |
| :--- | :--- | :--- |
| `@accelerate function ... end` | Defining reusable array code. This is the recommended default. | Arguments and returned values are protected. Non-returned locals may fuse into consumers or be released after their last use. |
| `@accelerate begin ... end` | Named results must remain in the current scope. | `begin` creates no new Julia scope, so every named binding is protected. An eligible same-shape CUDA chain may still use one multi-output kernel, but each named result is materialized. |
| `@accelerate let ... end` | Writing a one-off multi-statement calculation when only its result is needed. | `let` creates a local scope. Only the result escapes; other locals may fuse away or be released after their last use. |
| `@accelerate expr` | Evaluating one expression without named intermediates. | The result is materialized and returned. Eligible operations fuse within the expression, and transient temporaries are released. |

For example, choose `begin` when both names are needed afterward:

```julia
@accelerate begin
    product = @. A * B
    shifted = @. product + 1
end

consume(product, shifted)
```

Choose `let` when only the final result should escape:

```julia
result = @accelerate let
    product = @. A * B
    @. product + 1
end
```

For a single unnamed expression, use:

```julia
result = @accelerate (@. A + B * C)
```

## Writing an accelerated body

- Apply `@.` to each elementwise right-hand side. Applying it to the entire body would change `x = ...` into `x .= ...` and `f(...)` into `f.(...)`.
- Use ordinary `=` for a disposable intermediate. This allows a single-use producer to fuse into its consumer.
- Use `.=` when the mutation must be visible. The destination write is preserved, although an eligible producer may fuse into it.
- Keep control flow outside the accelerated body. Loops, conditionals, `try`, short-circuit operators, and nested functions are rejected.
- Ordinary function calls run in program order and form rewrite boundaries. Annotate the called function separately if its body should also be accelerated.

```julia
for _ in 1:nsteps
    update!(C, A, B)
end
```

See [Kernel Fusion](./kernel_fusion.md) for CUDA fusion requirements and [`@show_lifetimes`](../debugging.md#inspect-lifetime-rewrites-with-show_lifetimes) to inspect the exact rewrite without executing it.
