# `@accelerate`

`@accelerate` rewrites straight-line array code to:

1. **Fuse across statements.** A temporary used once is folded into the
   statement that consumes it, so on a GPU the chain runs as one kernel.
2. **Free temporaries early.** Temporaries that can't be fused are freed right
   after their last use, instead of waiting for Julia's GC. This works on CPU
   too.

```julia
@accelerate function update!(C, A, B)
    tmp = @. A + B
    C .= @. 2.0f0 * tmp     # runs as C .= @. 2.0f0 * (A + B)
    return C
end
```

Arguments and returned values are never freed. Loops, conditionals, `try`,
`&&`/`||` and nested functions are rejected: put `@accelerate` on the function
called in the loop, not around the loop.

## Forms

| Form | What stays valid afterwards |
|:---|:---|
| `@accelerate function ... end` (recommended) | arguments and the returned value |
| `@accelerate let ... end` | the result only |
| `@accelerate begin ... end` | every named variable |
| `@accelerate expr` | the result |

```julia
result = @accelerate let
    product = @. A * B
    @. product + 1
end

result = @accelerate (@. A + B * C)
```

With `begin`, each named result is kept, but a chain of same-shape statements
can still run as one GPU kernel that writes all of them.

The form also affects how much memory is left for Julia's garbage collector to reclaim. In this Gray-Scott benchmark, the `function` and `let` forms have the lowest step times and leave no temporary arrays for the GC. The colors show the amount of memory left to the GC per GPU per step. `plain` is the version without `@accelerate`.

```@raw html
<object data="../images/benchmarks/grayscott_time_fused.pdf" type="application/pdf" style="width: 100%; aspect-ratio: 480 / 208;" aria-label="Gray-Scott step times for plain code and the expression, begin, let, and function forms of accelerate, colored by memory left to the garbage collector.">
  <a href="../images/benchmarks/grayscott_time_fused.pdf">Gray-Scott step times for plain code and the expression, begin, let, and function forms of accelerate, colored by memory left to the garbage collector.</a>
</object>
```

## Writing the body

- Put `@.` on each right-hand side, not on the whole body, which would turn
  `x = ...` into `x .= ...`.
- Use `=` for a throwaway temporary, so it can be fused away.
- Use `.=` when the write must happen, for example into an argument. A
  preallocated `tmp .= ...` is kept, not fused.
- Calls to other functions run in order and are not fused through.
  Annotate those functions separately if they should be accelerated too.

[`@show_lifetimes`](../debugging.md#inspect-lifetime-rewrites-with-show_lifetimes)
prints the rewritten code without running it.
