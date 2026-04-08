
# Public API

User facing functions supported by cuNumeric.jl

```@contents
Pages = ["api.md"]
Depth = 2:2
```

```@autodocs
Modules = [cuNumeric]
Pages = ["ndarray/ndarray.jl", "ndarray/unary.jl", "ndarray/binary.jl", "cuNumeric.jl", "warnings.jl", "util.jl", "memory.jl", "scoping.jl"]
```

# CUDA.jl Tasking

Write custom GPU kernels in Julia using CUDA.jl and execute them through the Legate distributed runtime. Your kernels automatically benefit from Legate's data partitioning, dependency tracking, and multi-GPU scheduling.

!!! warning "Experimental Feature"
    CUDA.jl tasking is experimental. You must opt in before using `@cuda_task` or `@launch`:
    ```julia
    cuNumeric.Experimental(true)
    ```

The interface has two steps:
1. **Compile & Register** — [`@cuda_task`](@ref) JIT-compiles a kernel to PTX and registers it with Legate.
2. **Launch** — [`@launch`](@ref) submits the kernel with grid dimensions, inputs, outputs, and scalars.

`NDArray` arguments are automatically mapped to their CUDA equivalents (`NDArray{T,1}` → `CuDeviceVector{T,1}`, etc.). Scalar arguments are passed through by copy.

!!! note "Argument ordering"
    Legate passes kernel arguments in the order: **inputs → outputs → scalars**.
    Your kernel signature must match this ordering.

!!! warning "Inputs vs. outputs"
    Correctly separating `inputs` and `outputs` is critical for Legate's
    dependency analysis. If an array is both read and written, list it as an `output`.

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

# 2. Compile & register — args are used only for type inference
task = cuNumeric.@cuda_task kernel_sin(a, b, UInt32(1))

# 3. Launch through Legate
cuNumeric.@launch task=task threads=threads blocks=blocks inputs=a outputs=b scalars=UInt32(N)

allowscalar() do
    println("sin(1) = ", b[:][1])  # ≈ 0.8414709
end
```

See `examples/custom_cuda.jl` for a more complete example with multiple kernels.

## `@launch` Keywords

| Keyword   | Type                 | Default  | Description                    |
|-----------|----------------------|----------|--------------------------------|
| `task`    | `CUDATask`           | required | Compiled kernel handle         |
| `blocks`  | `Int` or `Tuple`     | `(1,)`   | CUDA grid dimensions           |
| `threads` | `Int` or `Tuple`     | `(256,)` | CUDA block dimensions          |
| `inputs`  | `NDArray` or `Tuple` | `()`     | Read-only input arrays         |
| `outputs` | `NDArray` or `Tuple` | `()`     | Read-write output arrays       |
| `scalars` | scalar or `Tuple`    | `()`     | Scalar kernel arguments        |

## Limitations

- Only `NDArray` objects are supported — raw `CuArray` cannot be passed directly.
- Mismatched array sizes are automatically padded to the largest shape.
- Custom function broadcasting is not supported; write explicit index-based kernels.

## API Reference

```@autodocs
Modules = [cuNumeric]
Pages = ["cuda.jl"]
```

# CNPreferences

This section details how to set custom build configuration options. To see more details visit our install guide [here](./install.md).

```@autodocs
Modules = [CNPreferences]
Pages = ["CNPreferences.jl"]
```

# Internal API

```@autodocs
Modules = [cuNumeric]
Pages = ["ndarray/detail/ndarray.jl"]
```
