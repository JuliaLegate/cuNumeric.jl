# Benchmark configuration

Benchmarks are declared in `benchmarks.toml`. `run.jl` parses it.

## Running

```bash
julia --project=. run.jl   # runs whatever benchmarks.toml configures
```

`run.jl` runs each (benchmark, backend) pair in its own process via
`run_benchmark.sh`, so backends never share a GPU/runtime within a measurement.
cuNumeric always runs; extra comparison backends are toggled in `[Global]`:

- `cuda = true` → also run under CUDA.jl (single-GPU configs only; CUDA.jl is
  single-device). Every kernel has a `CuArray` implementation.
- `cupynumeric = true` → also run under cupynumeric (see below).

On a single GPU, the cuNumeric worker also compares a tiny problem against
CUDA.jl (`pass` / `fail` in the CSV). The timed CUDA.jl run still happens;
its correctness column is `skipped`. Multi-GPU and cupynumeric skip the check.

Individual `[[benchmark]]` blocks may override `cuda`, `n_warmup`, `n_iter`,
and `n_trial`. Unspecified values inherit from `[Global]`. This is useful for
enabling a single-GPU CUDA comparison only for compatible benchmarks or reducing
the iteration count for expensive kernels.

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

[[gemm]]            # name registered under src/benchmarks/
T    = "Float32"     # element type
gpus = 1
cpus = 2
N    = 150
M    = 150           # optional, defaults to 1
fusion = true        # optional, defaults to true; toggles cuNumeric broadcast fusion
```

Repeat a `[[name]]` block to add independent configs.

## Lists

Any of `T`, `fusion`, `gpus`, `cpus`, `N`, `M` may be a list. They expand along
two axes:
- **`T` and `fusion` multiply.** The whole sweep runs once per type and once per
  fusion setting (`fusion = [true, false]` sweeps both).
- **`gpus`, `cpus`, `N`, `M` zip** into a single lockstep sweep — element `i`
  of each is paired together.

`fusion` toggles cuNumeric broadcast fusion (`true`/`false` or `"on"`/`"off"`,
default `true`); it only affects cuNumeric, so comparison backends run once, not
per variant.

Benchmark names are defined by the registered benchmark implementations. A
benchmark may expose baseline, optimized, backend-specific, or other variants;
the harness treats each name uniformly and records each result independently.

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

## Tensor contractions

Two direct TensorOperations benchmarks compare the same mathematical kernel
across cuNumeric.jl, cuPyNumeric, and—when `cuda = true` on a one-GPU
entry—TensorOperations.jl's cuTENSOR backend:

- `tensor_projection3` computes
  `D[n,m,l] = A[i,j,k] * B[n,i] * B[m,j] * B[l,k]`. TensorOperations performs
  three pairwise contractions with rank-3 intermediates. For equal index extent
  `N`, the counted work is `3N^3(2N-1)`, asymptotically `6N^4`.
- `tensor_contract4` computes
  `C[a,b,c,d] = X[a,i,c,j] * Y[i,b,j,d]`. This single contraction isolates the
  primitive high-rank backend path and counts `N^4(2N^2-1)` operations.

The Julia implementations use `@tensor` on both `NDArray` and `CuArray`; the
latter activates TensorOperations' cuTENSOR extension and is recorded as
`TensorOperations.jl / cuTENSOR`. The cuPyNumeric implementations use equivalent
`einsum` expressions. Final outputs are preallocated, and contraction-order
selection is performed before timing (at macro expansion in Julia and during
initialization in Python). Required intermediate allocation and release remain
part of each timed projection iteration.

## Plotting

A full `run.jl` pass (no extra args) plots at the end. One-off CLI runs do
not. To plot existing CSVs:

```bash
julia --project=. plot_results.jl
```

`[plot.groups]` in `benchmarks.toml` puts related kernels on one figure.
Gray-Scott's baseline and `@accelerate` forms share `grayscott`; DMD baseline
and accelerated share `dmd`. Every other `[[benchmark]]` table is its own
figure. Each figure overlays CUDA.jl (1 GPU) and cupynumeric from that
group's baseline CSV (`*_baseline`, else the only / first name). Accelerated
kernels have no CUDA.jl or Python CSVs; the overlay still comes from the
baseline.

cuNumeric fused vs unfused uses the same color with solid vs dashed lines.
Outputs are `plots/<group>_weak_scaling.png`. Optional flags: `--out=`,
`--suffix=`, `--config=`, or a results-directory path.
