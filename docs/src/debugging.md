# Debugging

Debug the layer that matches the problem:

| Question | Tool |
|---|---|
| Which operations did Legate submit, and when did they run? | [Legate logs and profiles](#trace-legate-runtime-work) |
| How were broadcasts fused? | [`BCAST_FUSION_DEBUG`](#inspect-fused-broadcasts-with-bcast_fusion_debug) |
| How does `@accelerate` rewrite code and free temporaries? | [`@show_lifetimes`](#inspect-lifetime-rewrites-with-show_lifetimes) |

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

With task-scope naming enabled and Julia restarted, cuNumeric operations carry short labels such as `zeros`, `matmul`, and
`broadcast.+(*(input{0}, input{1}), 2.0f0)`. Search for those labels in `legate_*.log` to connect runtime messages to calls in `workload.jl`.

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
`@accelerate` expands, so enable the flag before defining or evaluating
the expression you want to inspect. Kernel details are reported at runtime.

```julia
using cuNumeric

cuNumeric.BCAST_FUSION_DEBUG[] = true

N = 10
A = cuNumeric.ones(Float32, N, N)
B = cuNumeric.ones(Float32, N, N)
C = cuNumeric.zeros(Float32, N, N)

@accelerate function combine!(C, A, B)
    product = A[2:end-1, 2:end-1] .* B[2:end-1, 2:end-1]
    C[2:end-1, 2:end-1] = product .+ 2.0f0
    return C
end

cuNumeric.BCAST_FUSION_DEBUG[] = false
```

For example, a single-use producer inside `@accelerate` is reported as:

```text
======================================== inter-broadcast fusion rewrite
  before
    begin
        product = A[2:end - 1, 2:end - 1] .* B[2:end - 1, 2:end - 1]
        C[2:end - 1, 2:end - 1] = product .+ 2.0f0
    end
  fused
    C[2:end - 1, 2:end - 1] .= A[2:end - 1, 2:end - 1] .* B[2:end - 1, 2:end - 1] .+ 2.0f0
```

`before` contains exactly the statements that were recombined, and `fused`
contains their replacement. No rewrite block is printed when the pass leaves
the statements unchanged.

The fused kernel is reported separately:

```text
======================================== fused broadcast kernel
  expr    +(*(input{0}, input{1}), 2.0f0)
  output  NDArray{Float32, 2} (8, 8) slice, parent (10, 10)
  inputs  input{N} (2 unique)
    N   value
    0   NDArray{Float32, 2} (8, 8) slice, parent (10, 10)
    1   NDArray{Float32, 2} (8, 8) slice, parent (10, 10)
  launch  host thread budget=1024, indexing=cartesian, blocks=device(local tile), global_ndrange=(8, 8)
  call    broadcast.gpu_broadcast_kernel_cartesian_splat(input{0}, input{1}, 2.0f0)
```

How to read it:

- `expr` is the fused op tree. Its `input{N}` names index the `inputs` table;
  scalar arguments appear directly as their runtime values.
- `inputs` summarizes each runtime array without exposing internal `NDArray`
  type parameters. A slice includes its parent shape. Source names such as `A`
  are available in the static lifetime rewrite, but Julia does not retain those
  binding names on the runtime array object.
- `call` shows the packed kernel argument order, including reused inputs.
- If no kernel block prints, the expression took the unfused path (for example shape-mismatched leaves, fusion disabled, or below the min-ops threshold).

Turn the flag off when you are done. It prints on every fused launch and can be noisy in loops.

See [Kernel Fusion](./perf/kernel_fusion.md) for `@.` / fusion usage, and [Internals](./internals.md) for how these features work.

## Inspect lifetime rewrites with `@show_lifetimes`

`@accelerate` rewrites straight-line code so eligible broadcasts combine and non-returned temporaries are freed after their final use. `@show_lifetimes` prints the exact expansion without executing it, so it works without a GPU.

```julia
using cuNumeric

@show_lifetimes function update!(C, A, B)
    result = A[1:end, :] .+ B[1:end, :]
    C .= result .* 2.0
    return C
end
```

How to read it:

- The header identifies the exact function, `let`, block, or expression form expanded.
- Numbered lines are rewritten statements.
- Red `✗ free tmpN` lines are the inserted `maybe_insert_delete` calls.
- With fusion enabled, dotted intermediates stay as broadcast expressions instead of being treated as separate allocations. With fusion disabled, the header says `plain` and more call sites are hoisted.

Use this when a hot loop still looks allocation-heavy, or when you want to confirm that a value is freed before it escapes the block.
