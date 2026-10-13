# Internals

How [kernel fusion](./perf/kernel_fusion.md), [`@accelerate`](./perf/accelerate.md)
and memory management work under the hood.

## Broadcast fusion

Julia builds a nested `Broadcasted` tree for a dotted expression. If fusion is
enabled, a GPU is present and the tree is eligible, cuNumeric flattens it,
generates a CUDA.jl kernel and launches it as a Legate task. Otherwise each
operation dispatches as a single cuPyNumeric operation in the Legate runtime.

Usage: [Kernel Fusion](./perf/kernel_fusion.md). Source: `src/ndarray/broadcast.jl`,
`src/ndarray/broadcast_fusion.jl`.

## `@accelerate`

`@accelerate` rewrites code before it runs. Take:

```julia
@accelerate function f!(C, A, B)
    tmp = A .+ B          # a temporary, used once
    C .= tmp .* 2
    return C
end
```

1. **Vertical fusion.** `tmp` is used only once, so it is substituted into the
   next line: `C .= (A .+ B) .* 2`. Two kernels become one and `tmp` is never
   allocated.
2. **Early frees.** Temporaries that can't be substituted (used twice, or
   fusion is off) get a free call inserted right after their last use.
   Arguments and returned values are never freed.

`@show_lifetimes` prints the rewritten code.

Usage: [`@accelerate`](./perf/accelerate.md). Source: `src/scoping/`.

## Memory and GC

An `NDArray` is a small Julia object pointing at storage owned by Legate, so
Julia's GC doesn't see how much memory arrays use. cuNumeric tracks the bytes
it allocates and frees. When that total exceeds 80% (soft limit) or 90% (hard
limit) of device or host memory, it triggers Julia's GC. `cuNumeric.disable_gc!()`
turns this off.

To free temporaries without waiting for the GC, see
[`@accelerate`](./perf/accelerate.md). Source: `src/memory.jl`.
