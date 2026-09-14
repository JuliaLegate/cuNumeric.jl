# CUDA.jl Tasking

Write custom GPU kernels in Julia using CUDA.jl and execute them through the Legate distributed runtime. Your kernels automatically benefit from Legate's data partitioning, dependency tracking, and multi-GPU scheduling.

!!! warning "Experimental Feature"
    CUDA.jl tasking is experimental. You must opt in before using `@cuda_task` or `@launch`:
    ```julia
    cuNumeric.Experimental(true)
    ```

The interface has two steps:
1. **Compile & Register** - `@cuda_task` JIT-compiles a kernel to PTX and registers it with Legate.
2. **Launch** - `@launch` submits the kernel with grid dimensions, inputs, outputs, and scalars.

`NDArray` arguments are automatically mapped to their CUDA equivalents (`NDArray{T,1}` → `CuDeviceVector{T,1}`, etc.). Scalar arguments are passed through by copy.

!!! warning "Inputs vs. outputs"
    Correctly separating `inputs` and `outputs` is critical for Legate's
    dependency analysis. If an array is both read and written, list it as an `output`.

!!! warning "Array sizes"
    Mismatched array sizes are automatically padded to the largest shape. To address this, we plan to add support for other Legate constraints in the future (more information [here](https://docs.nvidia.com/legate/latest/api/cpp/generated/group/group__partitioning.html)).

## Example

```julia
using cuNumeric
using CUDA
import CUDA: i32

# Enable experimental features
cuNumeric.Experimental(true)

# 1. Write a standard CUDA.jl kernel
function kernel_sin(a, b, N)
    i = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    if i <= N
        @inbounds b[i] = sin(a[i])
    end
    return nothing
end

N = 1024
threads = 256
blocks = cld(N, threads)

a = cuNumeric.fill(1.0f0, N)
b = cuNumeric.zeros(Float32, N)

# 2. Compile and register (args are used only for type inference)
task = cuNumeric.@cuda_task kernel_sin(a, b, UInt32(1))

# 3. Launch through Legate
cuNumeric.@launch task=task threads=threads blocks=blocks inputs=a outputs=b scalars=UInt32(N)

allowscalar() do
    println("sin(1) = ", b[:][1])  # ≈ 0.8414709
end
```

See `examples/custom_cuda.jl` for a runnable two-kernel example.

## API Reference

```@autodocs
Modules = [cuNumeric]
Pages = ["cuda/cuda_ptx_task.jl"]
```
