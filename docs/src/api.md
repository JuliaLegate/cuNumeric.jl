
# Public API

User facing functions supported by cuNumeric.jl

```@contents
Pages = ["api.md"]
Depth = 2:2
```

### Supported Unary Operations
The following unary operations are supported and can be broadcast over `NDArray`:

  • `-`, `!`, `abs`, `acos`, `acosh`, `asin`, `asinh`, `atan`, `atanh`, `cbrt`, `conj`, `cos`, `cosh`, `deg2rad`, `exp`, `exp2`, `expm1`, `floor`, `imag`, `isfinite`, `log`, `log10`, `log1p`, `log2`, `rad2deg`, `real`, `sign`, `signbit`, `sin`, `sinh`, `sqrt`, `tan`, `tanh`, `^2`, `^-1` or `inv`,

##### Differences
- The `acosh` function in Julia will error on inputs outside of the domain (x >= 1)
    but cuNumeric.jl will return NaN.



### Supported Binary Operations
The following binary operations are supported and can be applied elementwise to pairs of `NDArray` values:

  • `+`, `-`, `*`, `/`, `^`, `<`, `<=`, `>`, `>=`, `==`, `!=`, `atan`, `hypot`, `max`, `min`, `lcm`, `gcd`

These operations are applied elementwise by default and follow standard Julia semantics.


```@autodocs
Modules = [cuNumeric]
Pages = ["ndarray/ndarray.jl", "ndarray/unary.jl", "ndarray/binary.jl", "cuNumeric.jl", "warnings.jl", "util.jl", "memory.jl", "scoping.jl"]
```

# CNPreferences

This section details how to set custom build configuration options. To see more details visit our install guide [here](./install.md).

```@autodocs
Modules = [CNPreferences]
Pages = ["CNPreferences.jl"]
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

# 2. Compile & register — args are used only for type inference
task = cuNumeric.@cuda_task kernel_sin(a, b, UInt32(1))

# 3. Launch through Legate
cuNumeric.@launch task=task threads=threads blocks=blocks inputs=a outputs=b scalars=UInt32(N)

allowscalar() do
    println("sin(1) = ", b[:][1])  # ≈ 0.8414709
end
```

See `examples/custom_cuda.jl` for a more complete example with multiple kernels.

## API Reference

```@autodocs
Modules = [cuNumeric]
Pages = ["utilities/cuda_stubs.jl"]
```

# Internal API

```@autodocs
Modules = [cuNumeric]
Pages = ["ndarray/detail/ndarray.jl"]
```
