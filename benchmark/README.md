# Benchmark configuration

Benchmarks are declared in `benchmarks.toml`. `run.jl` parses it.

## Layout

```toml
[Global]
n_warmup = 5
n_iter   = 1000
n_trial  = 5

[[sgemm]]            # name registered in src/benchmarks.jl
T    = "Float32"     # element type
gpus = 1
cpus = 2
N    = 150
M    = 150           # optional, defaults to 1
```

Repeat a `[[name]]` block to add independent configs.

## Lists

Any of `T`, `variants`, `gpus`, `cpus`, `N`, `M` may be a list. They expand along
two axes:

- **`T` and `variants` multiply.** The whole sweep runs once per type and once
  per variant.
- **`gpus`, `cpus`, `N`, `M` zip** into a single lockstep sweep — element `i`
  of each is paired together.

Each zipped field must be one of:

- a scalar or single-element list (`cpus = 2` or `[2]`) -> broadcast to every config
- a list whose length equals the sweep length

Any other length mismatch is an error.

```toml
[[sgemm]]
T    = ["Float64", "Float32"]   # multiplies
gpus = [1, 2, 4]                #
cpus = 2                        # zip -> (1,2,150,150), (2,2,300,300), (4,2,600,600)
N    = [150, 300, 600]          #
M    = [150, 300, 600]          #
```

-> 2 types * 3 sweep points = **6 runs**.

### Gotcha

When `T = ["Float32", "Float64"]` and a length-2 `N`/`M` sweep you get all **4**
combinations, not a paired `Float32 -> N[1], Float64 -> N[2]`. To pin a type
to a specific size, use separate `[[name]]` blocks.

## Variants

A variant is a named way of running a benchmark. List them per entry with
`variants = [...]` (defaults to `["baseline"]`); they multiply like `T`, and the
chosen variant is recorded as a column in the results CSV so runs can be compared.

```toml
[[grayscott]]
T = "Float64"
N = 1024
M = [1024, 2048, 4096]
gpus = [1, 2, 4]
cpus = 2
variants = ["baseline", "lifetimes"]   # 2 variants * 3 sweep points = 6 runs
```

There are two kinds, both flowing through the same `variant` string:

- **Code-path variants** change what the worker runs. The benchmark's `run!`
  dispatches on the variant. Example: grayscott's `lifetimes` wraps the step in
  `@analyze_lifetimes` (see `src/benchmarks/grayscott.jl`). A benchmark that
  doesn't recognize a variant just runs its baseline path.
- **Process-level variants** flip a runtime setting before the run via a setup
  thunk registered in `register_variant` (`src/core.jl`). The worker calls it at
  startup. Broadcast fusion will plug in here once it lands, e.g.
  `register_variant("fusion_off", cuNumeric.CNPreferences.disable_broadcast_fusion!)`.
