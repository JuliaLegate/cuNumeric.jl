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

The compiler parses the body once into operation and value records. A
`Graphs.SimpleDiGraph` connects producers to consumers and records ordering
constraints around effects. Operands remain ordered lists, so `f(x, x)` counts
as two uses. Reassigning a name creates a new value ID; aliases retain the old
ID. Generated locals keep both values addressable until their last use.

Fusion updates these records and their execution schedule. Under
`aggressive=true`, eligible adjacent indexed updates become a sibling launch.
Alias and read-freshness guards select between separate optimized and original
execution plans. Each plan receives its own cleanup placement.

A second directed graph connects lifetime parents to borrowing results, such
as an array and its lazy transpose. After fusion, the compiler propagates last
uses and escapes backward through this graph. It emits one deletion for each
remaining locally owned, nonescaping array. The graphs and binding maps exist
only during expansion; execution contains ordinary Julia operations, existing
alias guards, and deletion calls.

Unknown calls keep their results and inputs until scope exit. If their result
escapes, its inputs stay protected too. The compiler assumes that such calls
do not retain arguments through side effects; it does not inspect function
bodies. Borrowing declarations are internal implementation details, including
registrations supplied by package extensions.

For expansion measurements, run `julia --project scripts/benchmark_accelerate.jl`.
A Julia 1.10.12 comparison against `develop` (`8ddb229a`, fetched 2026-10-19)
used identical input syntax and stand-in backend types, with each compiler in
a separate process, five warmups and 30 samples. Minimum expansion time and
allocations for a four-operation chain changed from 0.52 ms / 172,280 bytes to
0.49 ms / 170,584 bytes. The Gray–Scott test function changed from
31.59 ms / 9,180,224 bytes to 16.18 ms / 5,132,488 bytes. These are expansion
measurements, not GPU execution benchmarks; timings depend on the environment.
The scalar accelerated function (`y = x + 1; y * 2`) allocates zero runtime
bytes on both develop and the graph compiler. The previously reported
1,040-byte saving was against the experimental ownership implementation on
this feature branch (`9dbda372`), not against develop.

A same-process Julia 1.13.1/A30 comparison evaluated both compilers' emitted
functions against the same backend. After warmup, ten alternating batches with
blocking execution fences measured a 100,000-element chain at 0.204 ms on
develop and 0.209 ms with the graph compiler; a 100×100 Gray–Scott step measured
2.306 ms and 2.286 ms respectively. Batches used 100 chain calls or 30 steps.
Julia allocations per call were 2,336 bytes for both chains, and 34,672 versus
35,728 bytes for Gray–Scott. This does not establish a material execution
speedup; the demonstrated performance benefit is expansion time.

Usage: [`@accelerate`](./perf/accelerate.md). Source: `src/scoping/`.

## Memory and GC

An `NDArray` is a small Julia object pointing at storage owned by Legate, so
Julia's GC doesn't see how much memory arrays use. cuNumeric tracks the bytes
it allocates and frees. When that total exceeds 80% (soft limit) or 90% (hard
limit) of device or host memory, it triggers Julia's GC. `cuNumeric.disable_gc!()`
turns this off.

To free temporaries without waiting for the GC, see
[`@accelerate`](./perf/accelerate.md). Source: `src/memory.jl`.
