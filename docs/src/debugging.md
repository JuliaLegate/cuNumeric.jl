# Debugging

cuNumeric.jl has two pretty printers that help you inspect what the compiler and runtime are doing.

## Inspect lifetime rewrites with `@show_lifetimes`

`@analyze_lifetimes` rewrites a block so temps are freed after their last use. `@show_lifetimes` prints the re-written code (without execution). It is pure AST work, so it works even without a GPU.

```julia
using cuNumeric

@show_lifetimes begin
    result = A[1:end, :] .+ B[1:end, :]
    C .= result .* 2.0
end
```

Example output when broadcast fusion is enabled (fusion-aware analysis):

```text
@analyze_lifetimes expansion (fusion-aware analysis)
------------------------------------------------------------
   1  tmp1 = A[1:end, :]
   2  tmp2 = B[1:end, :]
   3  tmp3 = tmp1 .+ tmp2
    ✗ free tmp1
    ✗ free tmp2
   4  result = tmp3
   5  res3 = (C .= result .* 2.0)
    ✗ free tmp3
   6  res3
------------------------------------------------------------
```

How to read it:

- Numbered lines are the rewritten statements.
- Red `✗ free tmpN` lines are the inserted `maybe_insert_delete` calls.
- With fusion enabled, dotted intermediates stay as broadcast expressions instead of being treated as many separate allocations. With fusion disabled, the header says `plain analysis` and more call sites are hoisted.

Use this when a hot loop still looks allocation-heavy, or when you want to confirm that a value is freed before it escapes the block.

## Inspect fused broadcasts with `BCAST_FUSION_DEBUG`

When broadcast fusion is on, set `cuNumeric.BCAST_FUSION_DEBUG[] = true` to print each fused kernel before launch: the expression tree, inputs, scalars, arg map, and launch geometry.

```julia
using cuNumeric

cuNumeric.BCAST_FUSION_DEBUG[] = true

N = 8
A = cuNumeric.ones(Float32, N, N)
B = cuNumeric.ones(Float32, N, N)
C = cuNumeric.zeros(Float32, N, N)

C .= @. A * B + 2.0f0

cuNumeric.BCAST_FUSION_DEBUG[] = false
```

Example output:

```text
======================================== fused broadcast kernel
  expr    +(*(NDArray, NDArray), 2.0f0)
  output  NDArray{Float32, 2, false, Nothing} (8, 8)
  inputs  2 unique NDArray(s)
    [0] NDArray{Float32, 2, false, Nothing} (8, 8)
    [1] NDArray{Float32, 2, false, Nothing} (8, 8)
  scalars 2.0f0
  arg_map [0, 1, 2, -1]  (0=output, >=1=input idx+1, <0=scalar)
  launch  host thread budget=1024, indexing=linear, blocks=device(local tile), global_ndrange=(8, 8)
  call    broadcast.gpu_broadcast_kernel_linear_splat(input0, input1, scalar0)
```

How to read it:

- `expr` is the fused op tree.
- `inputs` / `scalars` are the runtime arguments passed into the kernel.
- `arg_map` encodes how kernel slots map to the output (`0`), inputs (`>= 1`), and scalars (`< 0`).
- If nothing prints, the expression took the unfused path (for example shape-mismatched leaves, fusion disabled, or below the min-ops threshold).

Turn the flag off when you are done. It prints on every fused launch and can be noisy in loops.

See [Kernel Fusion](./perf/kernel_fusion.md) for `@.` / fusion usage, and [Internals](./internals.md) for how these features work.

## Inspect Legate with logging and task-scope names

Legate itself is configured through `LEGATE_CONFIG` (set **before** Julia starts). For runtime logs, add logging flags to that string:

```bash
export LEGATE_AUTO_CONFIG=0
export LEGATE_CONFIG="--gpus 1 --cpus 4 --logging legate=debug --log-to-file"
julia --project=. -e 'using cuNumeric; ...'
```

What the flags do:

- `--logging legate=debug` raises the `legate` logger (levels include `info` / `debug` etc.).
- `--log-to-file` writes `legate_*.log` per rank under the launch directory (override with `--logdir <path>`).

For a timeline view instead of text logs, pass `--profile` (produces `legate_*.prof`; view with `legate_prof`). Prefer logging **or** profiling in a given run; mixing them can distort timings.

### Named task scopes in those logs

You can label regions of Julia code with `Legate.with_scope(name) do ... end`. The string becomes provenance on operations submitted inside the block, which shows up in `legate_*.log` (or a `--profile` trace) and makes it easier to map runtime work back to your program.

Start Julia with logging (or `--profile`) already set in `LEGATE_CONFIG`, then wrap the phases you care about:

```bash
export LEGATE_CONFIG="--gpus 1 --cpus 4 --logging legate=debug --log-to-file"
julia --project=.
```

```julia
using cuNumeric
import Legate

A = Legate.with_scope("setup") do
    cuNumeric.ones(Float32, 64, 64)
end
B = Legate.with_scope("setup") do
    cuNumeric.ones(Float32, 64, 64)
end

D = Legate.with_scope("fused_broadcast") do
    @. A * B + 2.0f0
end
```

Separately, [task scope names](./api_preferences.md#task-scope-names) (`CNPreferences.enable_task_scope_names!`) is a **compile-time** preference that auto-wraps many cuNumeric entry points in `Legate.with_scope` with short op labels (for example `matmul`, `zeros`, or `broadcast.+(*(input0, input1), scalar0)`). That is optional and independent of manual `with_scope` blocks: flip it, restart Julia, and individual ops get named even without wrapping your own code. Call `CNPreferences.disable_task_scope_names!()` and restart when you are done.
