# Benchmarking

The code and configurations are in our [benchmarking repository](https://github.com/JuliaLegate/benchmarking). Run `./run_all.sh` from the repository root, or read its [README](https://github.com/JuliaLegate/benchmarking#readme) for setup and individual benchmarks.

## Package Integrations

The [Krylov.jl CG](../examples/krylov.md) and [OrdinaryDiffEq.jl heat diffusion](../examples/ordinarydiffeq.md) benchmarks compare cuNumeric.jl, CUDA.jl, and Dagger.jl arrays. The left panels vary the problem size on one GPU. The right panels show weak scaling, where cuNumeric.jl keeps solve and step times nearly constant as the GPU count grows.

Lower times are better. Click the figure to open the PDF.

[![Krylov.jl CG solve times and OrdinaryDiffEq.jl heat diffusion step times on one and multiple GPUs.](../images/benchmarks/composability_cunumeric.png)](../images/benchmarks/composability.pdf)

## cuNumeric Kernels and Applications

These benchmarks are written entirely with cuNumeric.jl and compared against cuPyNumeric and other Julia GPU frameworks. Lower times are better. Click a figure to open the PDF.

### Weak Scaling

Time per step on 1–8 GPUs for each benchmark, compared with cuPyNumeric, CUDA.jl, Dagger.jl, and JACC.jl.

[![Weak scaling time per step for GEMM, Monte Carlo, Gray-Scott, HPC-CG, NAS EP, NAS FT, and NAS MG.](../images/benchmarks/grid_time.png)](../images/benchmarks/grid_time.pdf)

### Broadcast Fusion

Gray-Scott and Monte Carlo with and without broadcast fusion, compared with cuPyNumeric and ImplicitGlobalGrid.jl.

[![Gray-Scott and Monte Carlo time per step with and without broadcast fusion.](../images/benchmarks/fusion_time.png)](../images/benchmarks/fusion_time.pdf)

### Gray-Scott

Throughput and time per step for the 2D Gray-Scott stencil, compared with cuPyNumeric and ImplicitGlobalGrid.jl.

[![Gray-Scott throughput and time per step on one and multiple GPUs.](../images/benchmarks/grayscott_compare.png)](../images/benchmarks/grayscott_compare.pdf)

### Problem Size Sweep

Time per step on one GPU as the problem size grows. At small sizes, fixed per-task runtime overhead dominates and hand-written CUDA.jl kernels are faster. The gap closes as the problem grows.

[![Gray-Scott and Monte Carlo time per step versus problem size on one GPU.](../images/benchmarks/sweep_time.png)](../images/benchmarks/sweep_time.pdf)

### Code Complexity vs Performance

Cyclomatic complexity of each implementation against its performance relative to the fastest model on 8 GPUs. Small markers are individual benchmarks and large markers are per-model means. The ideal is the top-left corner.

[![Cyclomatic complexity versus relative performance on 8 GPUs for cuNumeric.jl, Dagger.jl, and JACC.jl.](../images/benchmarks/complexity_8gpu.png)](../images/benchmarks/complexity_8gpu.pdf)
