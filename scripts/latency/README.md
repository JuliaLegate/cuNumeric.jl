# `@accelerate` latency

Run from an environment that resolves the cuNumeric checkout you want to measure.
Instantiate and precompile its dependencies first; installation and dependency
precompilation are separate from these package-level measurements.

```sh
julia --project=ENV -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  julia --startup-file=no --project=ENV scripts/latency/measure_accelerate.jl 3
```

Set the usual `LEGATE_CONFIG` and fusion preferences before preparing the
environment. The driver preserves them and starts a fresh Julia process for
each sample. It prints individual samples followed by medians and ranges.
Select fewer cases by listing them after the sample count:

```sh
julia --project=ENV scripts/latency/measure_accelerate.jl 3 precompile function stencil
```

- `precompile`: force compilation of cuNumeric with `Base.compilecache`, with
  dependency caches already available and `LEGATE_SKIP_RUNTIME=true`. This
  includes inference, code generation, and writing the package cache. It is not
  the time to install and precompile the entire dependency graph.
- `load`: `using cuNumeric`, including its normal runtime initialization, with
  an existing cache. Julia process startup is outside this timer.
- `expand_first` / `expand_warm`: first and warmed macro expansion, excluding
  execution of the generated code. Each timed expansion includes a fresh
  `deepcopy` of the quoted input so mutation by a rewrite cannot affect later
  samples. Each operation case reports these metrics.
- `definition`: evaluation/lowering of the already-expanded function definition.
- `inputs`: construction of the small input arrays and the initial blocking fence.
- `execute_first` / `execute_warm`: first and warmed calls, including a blocking
  execution fence. First execution includes remaining Julia and GPU compilation.
- `first_use`: expansion + definition + first execution. Load and input setup
  are reported separately.

The cases cover all four scope forms, a multi-output chain, a 2D sliced stencil,
and a plain broadcast control. Inputs are 32×32 Float32 arrays. Each process
checks results against ordinary Julia arrays before measuring warmed execution;
the warm metric is the minimum of five calls. The compile-time submetrics come
from Julia's `@timed` when available (`NaN` on older Julia versions) and do not
account for every external compiler/runtime cost.

For expansion-only measurements without a GPU/runtime, set both
`LEGATE_SKIP_RUNTIME=true` and `LATENCY_EXPANSION_ONLY=true` and select operation
cases. This does not measure normal package startup or numerical execution.

For comparisons, use identical dependency versions, preferences, Julia versions,
thread counts, and runtime settings. Give each checkout its own first depot to
isolate cuNumeric caches, with a shared dependency depot later in
`JULIA_DEPOT_PATH`. Alternate baseline/candidate order and avoid concurrent
benchmarks. A fresh process still benefits from OS filesystem and driver caches.

## Measured tradeoff

Compared with develop `769b461b` on dubliner (2026-10-18), Julia 1.13.1,
Intel Xeon Gold 6238L, NVIDIA A30X, one Julia/BLAS thread, fusion enabled,
and `LEGATE_CONFIG='--cpus 1 --gpus 1 --utility 1 --fbmem 1024 --sysmem 1024'`:

| Metric | Develop | Syntax workload |
| --- | ---: | ---: |
| Package precompilation, dependencies cached | 50.48 s | 52.33 s |
| `using cuNumeric` | approximately 14.6 s | approximately 14.6 s |
| Function first use | 6.07 s | 3.35 s |
| `begin` first use | 5.64 s | 3.16 s |
| Multi-output first use | 5.53 s | 3.00 s |
| `let` first use | 5.44 s | 3.15 s |
| Expression first use | 4.97 s | 3.01 s |
| Sliced stencil first use | 6.67 s | 3.84 s |
| Plain broadcast first use (control) | 2.88 s | 2.90 s |

Operation and precompilation figures are medians of three fresh processes per
variant. Operation order alternated between variants. The operation matrix used
the same four quoted workloads inside a setup block; after simplifying that
wrapper to the final single compile block, precompilation was remeasured three
times and function/stencil first use was checked again (3.37 s / 3.87 s).
Dependencies and preferences were matched; these are package-only rebuild
measurements, not clean-install timings. The combined `.ji` and native cache
grew by approximately 1.56 MiB.

The improvement is in macro expansion: for the function case, 3.26 s became
0.49 s. First numerical execution still takes roughly 3 s and is not improved
by this syntax workload. Warm execution showed no consistent change. Package
loading remains the larger startup cost. Julia 1.10.12 was also checked for
correctness and compatibility; the timing comparison above is for 1.13.1.
