# Debugging

Debug the layer that matches the problem:

| Question | Tool |
|---|---|
| Which operations did Legate submit, and when did they run? | [Legate logs and profiles](#trace-legate-runtime-work) |
| How were broadcasts fused? | [`BCAST_FUSION_DEBUG`](#inspect-fused-broadcasts-with-bcast_fusion_debug) |
| Where does `@analyze_lifetimes` free temporaries? | [`@show_lifetimes`](#inspect-lifetime-rewrites-with-show_lifetimes) |

## Trace Legate runtime work

Legate records provenance on submitted operations. cuNumeric can provide that provenance automatically for individual operations, or you can add broader phase names manually.

Choose one output format for a run:

| Goal | `LEGATE_CONFIG` flags | Output |
|---|---|---|
| Read runtime decisions as text | `--logging legate=debug --log-to-file` | `legate_*.log` |
| Inspect an execution timeline | `--profile` | `legate_*.prof` |

Set `LEGATE_CONFIG` before Julia starts. Add `--logdir <path>` to keep the generated files outside the working directory. Logging and profiling add overhead, so collect them in separate runs when measuring performance.

### Label individual operations

Task-scope naming is off by default. Enable the compile-time preference in one Julia process:

```bash
julia --project=. -e \
  'using CNPreferences; CNPreferences.enable_task_scope_names!()'
```

Then start a fresh process with logging enabled:

```bash
export LEGATE_AUTO_CONFIG=0
export LEGATE_CONFIG="--gpus 1 --cpus 4 --logging legate=debug --log-to-file"
julia --project=. workload.jl
```

cuNumeric operations now carry short labels such as `zeros`, `matmul`, and `broadcast.+(*(input0, input1), scalar0)`. Search for those labels in `legate_*.log` to connect runtime messages to calls in `workload.jl`.

For a timeline instead, replace the logging flags with `--profile`, run the same workload, and process the resulting `legate_*.prof` files with `legate_prof`.

Disable the preference after debugging, then restart Julia:

```bash
julia --project=. -e \
  'using CNPreferences; CNPreferences.disable_task_scope_names!()'
```

See [Task scope names](./api_preferences.md#task-scope-names) for the preference API.

### Label larger phases

Use `Legate.with_scope` when phase names such as `initialize` and `update` are more useful than per-operation names:

```julia
using cuNumeric
import Legate

A, B = Legate.with_scope("initialize") do
    A = cuNumeric.ones(Float32, 64, 64)
    B = cuNumeric.ones(Float32, 64, 64)
    (A, B)
end

D = Legate.with_scope("update") do
    @. A * B + 2.0f0
end
```

cuNumeric already supplies names for individual operations when task-scope naming is enabled. Scopes can be nested, so you can wrap those operations in your own phase-level scopes, as shown above.

## Inspect fused broadcasts with `BCAST_FUSION_DEBUG`

When broadcast fusion is on, set `cuNumeric.BCAST_FUSION_DEBUG[] = true` to
print inter-statement rewrites and each fused kernel's expression tree,
arguments, and launch geometry. Inter-statement rewrites are reported when
`@analyze_lifetimes` expands, so enable the flag before defining or evaluating
the expression you want to inspect. Kernel details are reported at runtime.

```julia
using cuNumeric

cuNumeric.BCAST_FUSION_DEBUG[] = true

N = 8
A = cuNumeric.ones(Float32, N, N)
B = cuNumeric.ones(Float32, N, N)
C = cuNumeric.zeros(Float32, N, N)

@analyze_lifetimes begin
    product = A .* B
    C[:, :] = product .+ 2.0f0
end

cuNumeric.BCAST_FUSION_DEBUG[] = false
```

For example, a single-use producer inside `@analyze_lifetimes` is reported as:

```text
======================================== inter-broadcast fusion rewrite
  before
    begin
        product = A .* B
        C[:, :] = product .+ 2.0f0
    end
  fused
    C[:, :] .= A .* B .+ 2.0f0
```

`before` contains exactly the statements that were recombined, and `fused`
contains their replacement. No rewrite block is printed when the pass leaves
the statements unchanged.

The fused kernel is reported separately:

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
- If no kernel block prints, the expression took the unfused path (for example shape-mismatched leaves, fusion disabled, or below the min-ops threshold).

Turn the flag off when you are done. It prints on every fused launch and can be noisy in loops.

See [Kernel Fusion](./perf/kernel_fusion.md) for `@.` / fusion usage, and [Internals](./internals.md) for how these features work.

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
