using CUDA
using Random

const KERNEL_OFFSET = sizeof(CUDA.KernelState)

# cuNumeric.jl init will call this
function set_kernel_state_size()
    cuNumeric.register_kernel_state_size(UInt64(KERNEL_OFFSET))
end

function ndarray_to_cuda_dummy_arr(arg)
    if isa(arg, NDArray)
        T = cuNumeric.eltype(arg)
        # size = cuNumeric.size(arg)
        return CUDA.zeros(T, 0)
    elseif Base.isbits(arg)
        return arg
    else
        error("Unsupported argument type: $(typeof(arg))")
    end
end

function map_ndarray_cuda_type(arg)
    t = cuNumeric.ndarray_to_cuda_dummy_arr(arg)
    return typeof(CUDA.cudaconvert(t))
end

function map_ndarray_cuda_types(args...)
    converted = Any[]
    for arg in args
        push!(converted, cuNumeric.map_ndarray_cuda_type(arg))
    end
    return tuple(converted...)
end

function __to_stdvec_u32(v)
    sv = CxxWrap.StdVector{UInt32}()
    for x in v
        push!(sv, UInt32(x))
    end
    return sv
end

struct CUDATask
    func::String
    argtypes::NTuple{N,Type} where {N}
end

function Launch(kernel::CUDATask, inputs::Tuple{Vararg{cuNumeric.NDArray}},
    outputs::Tuple{Vararg{cuNumeric.NDArray}}, scalars::Tuple{Vararg{Any}}; blocks, threads)
    input_vec = cuNumeric.VectorNDArray()
    for arr in inputs
        cuNumeric.push_back(input_vec, CxxRef{cuNumeric.CN_NDArray}(arr.ptr))
    end
    output_vec = cuNumeric.VectorNDArray()
    for arr in outputs
        cuNumeric.push_back(output_vec, CxxRef{cuNumeric.CN_NDArray}(arr.ptr))
    end
    scalar_vec = Legate.VectorScalar()
    for s in scalars
        Legate.push_back(scalar_vec, Legate.Scalar(s))
    end

    cuNumeric.new_task(
        kernel.func, __to_stdvec_u32(blocks), __to_stdvec_u32(threads), input_vec, output_vec,
        scalar_vec,
    )
end

function launch(kernel::CUDATask, inputs, outputs, scalars; blocks, threads)
    Launch(kernel,
        isa(inputs, Tuple) ? inputs : (inputs,),
        isa(outputs, Tuple) ? outputs : (outputs,),
        isa(scalars, Tuple) ? scalars : (scalars,);
        blocks=isa(blocks, Tuple) ? blocks : (blocks,),
        threads=isa(threads, Tuple) ? threads : (threads,),
    )
end
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
   [`map_ndarray_cuda_types`](@ref cuNumeric.map_ndarray_cuda_types).
2. Using [`CUDA.code_ptx`](@ref) to compile the specified GPU kernel 
   (`f`) into raw PTX text for the inferred types.
3. Extracting the kernel's function symbol name from the PTX using 
   [`extract_kernel_name`](@ref cuNumeric.extract_kernel_name).
4. Registering the compiled PTX and kernel name with the Legate runtime 
   via [`ptx_task`](@ref cuNumeric.ptx_task), making it available for GPU execution.
5. Returning a [`CUDATask`](@ref cuNumeric.CUDATask) struct that stores 
   the kernel name and type signature, which can be used to configure 
   and launch the kernel later.

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
macro cuda_task(call_expr)
    fname = call_expr.args[1]
    fargs = call_expr.args[2:end]

    esc(quote
        local _buf = IOBuffer()
        local _types = $cuNumeric.map_ndarray_cuda_types($(fargs...))
        # generate ptx using CUDA.jl 
        CUDA.code_ptx(_buf, $fname, _types; raw=true, kernel=true)

        local _ptx = String(take!(_buf))
        local _func_name = cuNumeric.extract_kernel_name(_ptx)

        # issue ptx_task within legate runtime to register cufunction ptr with cucontext
        cuNumeric.ptx_task(_ptx, _func_name)

        # create a CUDAtask that stores some info for a launch config
        cuNumeric.CUDATask(_func_name, _types)
    end)
end
"""
    @launch(; task, blocks=(1,), threads=(256,), inputs=(), outputs=(), scalars=())

Launch a GPU kernel (previously registered via [`@cuda_task`](@ref))  through the Legate runtime.

# Keywords
- `task` — A [`CUDATask`](@ref cuNumeric.CUDATask) object, typically returned by [`@cuda_task`](@ref). 
- `blocks`  — Tuple or single element specifying the CUDA grid dimensions. Defaults to `(1,)`.
- `threads` — Tuple or single element specifying the CUDA block dimensions. Defaults to `(256,)`.
- `inputs`  — Tuple or single element of input NDArray objects.
- `outputs` — Tuple or single element of output NDArray objects.
- `scalars` — Tuple or single element of scalar values.

# Description
The `@launch` macro validates the provided keywords, ensuring only 
the allowed set (`:task`, `:blocks`, `:threads`, `:inputs`, `:outputs`, `:scalars`) 
are present. It then expands to a call to [`cuNumeric.launch`](@ref), 
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
macro launch(args...)
    allowed_keys = Set([:task, :blocks, :threads, :inputs, :outputs, :scalars])
    kwargs = Dict{Symbol,Any}()

    for ex in args
        if !(ex isa Expr && ex.head == :(=))
            error("All arguments must be keyword assignments, e.g. task=..., threads=...")
        end
        key = ex.args[1]
        val = ex.args[2]

        if !(key in allowed_keys)
            error("@launch macro received unexpected keyword: $(key)")
        end

        kwargs[key] = val
    end

    if !haskey(kwargs, :task)
        error("@launch macro requires 'task=...' to be provided.")
    end
    task = kwargs[:task]
    blocks = get(kwargs, :blocks, :((1)))
    threads = get(kwargs, :threads, :((256)))
    inputs = get(kwargs, :inputs, :(()))
    outputs = get(kwargs, :outputs, :(()))
    scalars = get(kwargs, :scalars, :(()))

    esc(
        quote
            cuNumeric.launch(
                $task, $inputs, $outputs, $scalars; blocks=($blocks), threads=($threads)
            )
        end,
    )
end
