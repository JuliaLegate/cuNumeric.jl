# The `@accelerate` Macro

`@accelerate` optimizes straight-line array code at macro-expansion time:

- On CPU and GPU, static last-use analysis releases non-returned temporary `NDArray`s as soon as they become dead.
- With CUDA broadcast fusion enabled, eligible dotted expressions can be combined into fewer GPU kernels.

Arguments are never freed, mutations remain visible, and returned values remain materialized. The macro has four forms:

```julia
@accelerate function f(args...)
    # reusable straight-line kernel
end

@accelerate begin
    # soft scope
end

@accelerate let
    # hard scope
end

result = @accelerate expr
```

## Choose a form

| Form | Use it when | What remains available afterward |
|---|---|---|
| `@accelerate function ... end` | Defining a reusable update or compute kernel. This is the recommended default. | Caller-owned arguments, their mutations, and returned values. |
| `@accelerate begin ... end` | Named results must remain in the surrounding scope. | Every named binding created by the block. |
| `@accelerate let ... end` | Writing a one-off multi-statement calculation whose intermediates should not escape. | Only the block result. |
| `@accelerate expr` | Accelerating one expression without introducing a new scope. | The materialized expression result. |

## 1. Function form

Use this form by default. Function arguments belong to the caller and are never freed. Non-returned locals may be fused into consumers or released after their final use.

```julia
@accelerate function update!(C, A, B)
    combined = @. A + B
    C .= @. 2.0f0 * combined
    return C
end
```

## 2. `begin` form

`begin` preserves normal Julia scope. Named bindings remain available after the block, so the macro cannot discard them. Eligible same-shape CUDA chains may still run as one multi-output kernel.

```julia
@accelerate begin
    product = @. A * B
    shifted = @. product + 1
end

# Both names are still defined here.
consume(product, shifted)
```

## 3. `let` form

`let` creates a hard scope. Only its result escapes, giving `@accelerate` freedom to fuse or release every non-returned intermediate.

```julia
result = @accelerate let
    product = @. A * B
    @. product + 1
end

# `product` is not defined here.
```

## 4. Expression form

Use the expression form when there are no named intermediates:

```julia
result = @accelerate (@. A + B * C)
```

The result is materialized before it is returned.

## Writing the body

Apply `@.` to each elementwise right-hand side. Do not wrap the entire `@accelerate` body in `@.`: block-wide dotting changes `x = ...` into `x .= ...` and changes an ordinary call such as `bc!(...)` into `bc!.(...)`.

Prefer ordinary assignment for a single-use intermediate:

```julia
temporary = @. A * B
C .= @. temporary + 1
```

An explicit in-place `.=` mutation is observable and therefore forms a fusion boundary. Ordinary function calls run in program order and also form boundaries; `@accelerate` does not rewrite inside a called function unless that function is separately annotated.

The body must be straight-line. Control flow (`if`, loops, `try`, short-circuit operators) and nested functions are rejected. Put control flow outside the accelerated region:

```julia
for _ in 1:nsteps
    update!(C, A, B)
end
```

See [Kernel Fusion](./kernel_fusion.md) for fusion requirements and [`@show_lifetimes`](../debugging.md#inspect-lifetime-rewrites-with-show_lifetimes) to inspect the exact rewrite without executing it.
