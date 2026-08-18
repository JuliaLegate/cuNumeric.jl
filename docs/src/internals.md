# Internals

This page summarizes the machinery behind [`@accelerate`](./perf/reduce_allocations.md), [kernel fusion](./perf/kernel_fusion.md), and allocation-driven garbage collection.

## Broadcast fusion

Julia first builds a nested `Broadcasted` tree for a dotted expression. When CUDA fusion is enabled and the tree is eligible, cuNumeric flattens it, generates a CUDA.jl kernel, and launches it through Legate. Otherwise `unravel_broadcast_tree` evaluates the operations individually. CPU execution always uses the unfused path.

`@accelerate` adds an inter-statement syntax pass. It can merge a single-use broadcast producer into its consumer when doing so preserves mutation and aliasing semantics. The soft-scope `begin` form instead keeps named values materialized and may lower an eligible chain to a multi-output CUDA kernel.

Relevant source: `src/ndarray/broadcast_fusion.jl` and `src/scoping/`.

## Lifetime analysis

An `NDArray` is a small Julia handle to storage owned by Legate, so Julia's heap pressure understates the size of live array data. `@accelerate` therefore performs static last-use analysis:

1. Expand nested `@.` macros and reject non-straight-line code.
2. Apply conservative inter-statement fusion where the selected macro form permits it.
3. Hoist materialized temporary values and find their final uses.
4. Insert `maybe_insert_delete` calls after those uses while protecting caller-owned arguments, named soft-scope bindings, and returned values.

With fusion enabled, nested dotted nodes remain lazy and are not counted as separate array allocations. With fusion disabled, allocating calls are analyzed individually. `@show_lifetimes` prints the exact expansion without running it.

The analysis is statement-linear rather than a control-flow graph pass. Apply it to straight-line function or loop bodies, not to control flow itself.

## Allocation-driven GC

Each `NDArray` reports its byte size when constructed and freed. When predicted live bytes cross soft and hard fractions of available memory—and enough new growth has accumulated—cuNumeric asks Julia to collect garbage. Eager last-use freeing reduces peak live storage; the GC heuristic covers allocations the static pass cannot prove dead.

Relevant source: `src/memory.jl`.
