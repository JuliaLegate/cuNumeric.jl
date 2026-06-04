# Benchmark configuration

Benchmarks are declared in `benchmarks.toml`. `run.jl` parses it.

## Running

```bash
julia --project run.jl     # runs whatever benchmarks.toml configures
```

`run.jl` runs each (benchmark, backend) pair in its own process via
`run_benchmark.sh`, so backends never share a GPU/runtime within a measurement.
cuNumeric always runs; extra comparison backends are toggled in `[Global]`:

- `cuda = true` → also run under CUDA.jl (single-GPU configs only; CUDA.jl is
  single-device).
- `cupynumeric = true` → also run under cupynumeric (see below).

### Comparing against cupynumeric

cupynumeric runs in a conda env whose major.minor matches this project's
resolved `cupynumeric_jll`. Build it once:

```bash
./install_cupynumeric.sh   # creates env cupynumeric-bench-<major.minor>
```

`run.jl` derives the env name automatically; override it with `CUPYNUMERIC_ENV`.

## Layout

```toml
[Global]
n_warmup = 5
n_iter   = 1000
n_trial  = 5

[[gemm]]            # name registered in src/benchmarks.jl
T    = "Float32"     # element type
gpus = 1
cpus = 2
N    = 150
M    = 150           # optional, defaults to 1
```

Repeat a `[[name]]` block to add independent configs.

## Lists

Any of `T`, `gpus`, `cpus`, `N`, `M` may be a list. They expand along
two axes:

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
