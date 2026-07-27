# Reduce Allocations

Every intermediate `NDArray` (from a slice, broadcast, or function call) allocates a fresh buffer and waits for the Julia GC to free it. Because the GC runs on memory pressure, many dead buffers accumulate and pressure cuNumeric's allocator.

`@analyze_lifetimes` performs a **static last-use analysis** at macro-expansion time and inserts eager `maybe_insert_delete` calls immediately after each temporary's final use. Freed buffers can then be reused by later same-sized allocations instead of waiting on GC.

When broadcast fusion is on, intermediate dotted nodes in a broadcast tree are not real `NDArray` allocations. The macro accounts for that automatically.

```julia
T = Float32
A = cuNumeric.ones(T, (N, N))
B = cuNumeric.ones(T, (N, N))
C = cuNumeric.zeros(T, (N, N))

@analyze_lifetimes begin
    result = @. A[1:end, :] + B[1:end, :]
    C .= @. result * 2.0f0
end
```

**Benchmark** (Gray-Scott reaction-diffusion, 512×512, 10 000 steps):

```
               user     system   elapsed   CPU    max RSS
without   106.50 s   23.87 s   58.66 s   222%   3786 MB
with       61.74 s   13.66 s   27.84 s   270%   2999 MB
```

~2× wall-clock speedup and ~800 MB lower peak memory with no algorithmic changes.

Use `@show_lifetimes` to print the rewrite without running it ([Debugging](../debugging.md)). For how the rewriter and GC heuristics work, see [Internals](../internals.md).
