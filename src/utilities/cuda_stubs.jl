## This file contains stubs for methods implemented in
## the CUDA package extensions not implemented 
## elsewhere in the package.

export @cuda_task, @launch

"""
    @cuda_task(f(args...))

Compile a Julia GPU kernel to PTX, register it with the Legate runtime, 
and return a `CUDATask` object for later launch.

# Arguments
- `f` — The name of the Julia CUDA.jl GPU kernel function to compile.
- `args...` — Example arguments to the kernel, used to determine the 
  argument type signature when generating PTX.

# Description
This macro automates the process of:
1. Inferring the CUDA argument types for the given `args` using 
   `map_ndarray_cuda_types`.
2. Using `CUDA.code_ptx` to compile the specified GPU kernel 
   (`f`) into raw PTX text for the inferred types.
3. Extracting the kernel's function symbol name from the PTX using 
   `extract_kernel_name`.
4. Registering the compiled PTX and kernel name with the Legate runtime 
   via `ptx_task`, making it available for GPU execution.
5. Returning a `CUDATask` struct that stores the kernel name and type signature,
   which can be used to configure and launch the kernel later.

# Notes
- The `args...` are not executed; they are used solely for type inference.
- This macro is intended for use with the Legate runtime and 
  assumes a CUDA context is available.
- Make sure your kernel code is GPU-compatible and does not rely on 
  unsupported Julia features.

# Example
```julia
mytask = @cuda_task my_kernel(A, B, C)
```
"""
macro cuda_task end

"""
    @launch(; task, blocks=(1,), threads=(256,), inputs=(), outputs=(), scalars=())

Launch a GPU kernel (previously registered via [`@cuda_task`](@ref))  through the Legate runtime.

# Keywords
- `task` — A `CUDATask` object, typically returned by [`@cuda_task`](@ref). 
- `blocks`  — Tuple or single element specifying the CUDA grid dimensions. Defaults to `(1,)`.
- `threads` — Tuple or single element specifying the CUDA block dimensions. Defaults to `(256,)`.
- `inputs`  — Tuple or single element of input NDArray objects.
- `outputs` — Tuple or single element of output NDArray objects.
- `scalars` — Tuple or single element of scalar values.

# Description
The `@launch` macro validates the provided keywords, ensuring only 
the allowed set (`:task`, `:blocks`, `:threads`, `:inputs`, `:outputs`, `:scalars`) 
are present. It then expands to a call to `cuNumeric.launch`, 
passing the given arguments to the Legate runtime for execution.

This macro is meant to provide a concise, declarative syntax for 
launching GPU kernels, separating kernel compilation (via `@cuda_task`) 
from execution configuration.

# Notes
- `task` **must** be a kernel registered with the runtime, usually from `@cuda_task`.
- All keyword arguments must be specified as assignments, e.g. `blocks=(2,2)` not positional arguments.
- Defaults are chosen for single-block, 256-thread 1D launches.
- The macro escapes its body so that the values of inputs/outputs/scalars are captured 
  from the surrounding scope at macro expansion time.

# Example
```julia
mytask = @cuda_task my_kernel(A, B, C)

@launch task=mytask blocks=(8,8) threads=(32,32) inputs=(A, B) outputs=(C)
```
"""
macro launch end