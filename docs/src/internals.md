# Internals

This page describes how two performance features work under the hood: broadcast fusion, and the memory management path built around `@analyze_lifetimes` and GC heuristics. For day-to-day usage tips, see [Kernel Fusion](./perf/kernel_fusion.md) and [Reduce Allocations](./perf/reduce_allocations.md).

## Broadcast fusion

Nested Julia broadcast expressions on `NDArray` can compile to a single CUDA PTX kernel instead of launching one cuNumeric op per node in the expression tree. Fusion is automatic when it is enabled (the default). Users turn it on or off through `CNPreferences`; see [Configuration](./configuration.md).

### Pipeline

1. You write a dotted or `@.` expression such as `y .= @. a * b + c`.
2. Julia builds a lazy `Broadcasted` tree.
3. `copyto!(::NDArray, bc)` decides whether to fuse or fall back.
4. **Fused path:** promotion checks, flatten the tree, split runtime vs static args, build a KernelAbstractions linear kernel, compile or reuse PTX, launch through Legate's `RUN_PTX_BROADCAST` task.
5. **Unfused path:** `unravel_broadcast_tree` materializes intermediate `NDArray`s and calls one C-API ufunc per node.

```julia
# Eligible for fusion (nested ops, same shapes, CUDA available)
y .= @. a * b + c / d

# Typical unfused fallback: shape-mismatched broadcast such as matrix .+ vector
# (or fusion disabled / below the min-ops threshold)
```

### Design points

- Fusion lives inside Julia's broadcast `copyto!` path. It is not a general compiler pass over arbitrary Julia code.
- Only **linear, same-shape** array leaves fuse. Mismatched shapes fall back to the unfused path.
- The min-ops preference (default `2`) keeps trivial single-op broadcasts like `y .= cos.(x)` on the mature C-API path, avoiding PTX compile cost. Set it to `1` via `CNPreferences.set_broadcast_fusion_min_ops!(1)` if you want those fused too (restart Julia afterward).
- Preference changes are compile-time. Restart Julia after changing them. The ENV override `CUNUMERIC_FUSE_BROADCAST_MIN_OPS` is documented on the [Configuration](./configuration.md) page; prefer `CNPreferences` for normal use.
- For debugging fused launches and lifetime rewrites, see [Debugging](./debugging.md).

Relevant source: `src/ndarray/broadcast.jl`, `src/ndarray/broadcast_fusion.jl`.

## Lifetimes and GC

Julia's GC sees an `NDArray` as a small handle. The real cost lives in Legion / CUDA buffers. cuNumeric.jl therefore uses two cooperating layers: eager last-use freeing, and allocation-driven GC heuristics.

### Eager last-use freeing with `@analyze_lifetimes`

`@analyze_lifetimes` rewrites a block at macro-expansion time:

1. Hoist temporary allocations into named temps.
2. Find each temp's static last use.
3. Insert `maybe_insert_delete` (which calls `destroy!` for `NDArray`s) right after that last use.

Under broadcast fusion, intermediate dotted nodes are **not** real `NDArray` allocations. The macro switches to a fusion-aware hoist that keeps dotted trees lazy and only treats slices, broadcast roots, and non-broadcast calls as real allocations. When fusion is off, every call (including dotted ops) is treated as a real allocation.

```julia
@analyze_lifetimes begin
    result = A[1:end, :] .+ B[1:end, :]
    C .= result .* 2
end
```

Use `@show_lifetimes` to print the rewritten block and the free sites without running the code. That is pure AST work and works without a GPU.

Limits to keep in mind:

- Analysis is statement-linear. It is not a full control-flow graph pass. Wrap hot loop bodies, not entire programs.
- Some paths free eagerly outside the macro (for example LHS slice views created during indexed assignment).

Relevant source: `src/scoping.jl`.

### Allocation-driven GC heuristics

Every `NDArray` registers its byte size on construct and free. When predicted live bytes cross soft (~80%) or hard (~90%) fractions of available Legion memory, and enough new growth has accumulated since the last collection (hysteresis), cuNumeric triggers Julia `GC.gc`. Afterward it recalibrates counters from Legion's allocated-byte queries.

```text
on allocate:
  if live bytes > hard limit (~90%): full GC, then recalibrate
  else if live bytes > soft limit (~80%): incremental GC, then recalibrate
  else: allow the next spike to fire without waiting on the post-GC floor
```

`@analyze_lifetimes` reduces peak live temps. The heuristics catch cases the macro cannot see. You can disable the heuristics with `cuNumeric.disable_gc!()` (used in some tests).

Relevant source: `src/memory.jl`.
