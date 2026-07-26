# Benchmark Results

For JuliaCon2025 we benchmarks cuNumeric.jl on 8 A100 GPUs (single-node) and compared it to the Python library cuPyNumeric and other relevant benchmarks depending on the problem. All results shown are weak scaling. We hope to have multi-node benchmarks soon!

## SGEMM

Code Outline:
```julia
mul!(C, A, B)
```

```@raw html
<table>
  <tr>
    <th>GEMM Efficiency</th>
    <th>GEMM GFLOPS</th>
  </tr>
  <tr>
    <td><img src="../images/gemm_efficiency.svg" alt="GEMM Efficiency"/></td>
    <td><img src="../images/gemm_gflops.svg" alt="GEMM GFLOPS"/></td>
  </tr>
</table>
```

## Monte-Carlo Integration

Monte-Carlo integration is embaressingly parallel and should scale perfectly. We do not know the exact number of operations in `exp` so the GFLOPs is off by a constant factor.

Code Outline:
```julia
integrand = (x) -> exp.(-x.^2)
val = (V/N) * sum(integrand(x))
```

```@raw html
<table>
  <tr>
    <th>MC Efficiency</th>
    <th>MC GFLOPS</th>
  </tr>
  <tr>
    <td><img src="../images/mc_eff.svg" alt="MC Efficiency"/></td>
    <td><img src="../images/mc_ops.svg" alt="MC GFLOPS"/></td>
  </tr>
</table>
```

## Gray-Scott (2D)

Solving a PDE requires halo exchanges and lots of data movement. In this benchmark we fall an order of magnitude short of the `ImplicitGlobalGrid.jl` library which specifically targets multi-node, multi-GPU halo exchanges. Broadcast fusion helps on the elementwise update, but communication and stencil data movement still dominate the gap.

```@raw html
<table>
  <tr>
    <th>GS GFLOPS</th>
  </tr>
  <tr>
    <td><img src="../images/gs_gflops_diffeq.svg" alt="GS GFLOPS"/></td>
  </tr>
</table>
```
