using CUDA
using Random

const KERNEL_OFFSET = sizeof(CUDA.KernelState)

# cuNumeric.jl init will call this
function set_kernel_state_size()
    cuNumeric.register_kernel_state_size(UInt64(KERNEL_OFFSET))
end

function map_ndarray_cuda_type(arg)
    if isa(arg, NDArray)
        T = cuNumeric.eltype(arg)
        D = cuNumeric.ndims(arg)
        return CuDeviceArray{T,D,1}
    elseif Base.isbits(arg)
        return typeof(arg)
    else
        error("Unsupported argument type: $(typeof(arg))")
    end
end

function map_ndarray_cuda_types(args...)
    converted = Any[]
    for arg in args
        push!(converted, cuNumeric.map_ndarray_cuda_type(arg))
    end
    return tuple(converted...)
end

function to_stdvec(::Type{T}, vec) where {T}
    stdvec = CxxWrap.StdVector{T}()
    for x in vec
        push!(stdvec, T(x))
    end
    return stdvec
end

struct CUDATask
    func::String
    argtypes::NTuple{N,Type} where {N}
end

function add_padding(arr::NDArray, dims::Dims{N}; copy=false) where {N}
    old_size = size(arr)

    @assert all(dims .>= old_size) "newdims must be ≥ current dims elementwise"
    new = zeros(eltype(arr), dims)

    if copy # due to being an input. we don't need to copy outputs
        indices = ntuple(d -> 1:old_size[d], length(old_size))
        assign(new[indices...], arr)
    end

    nda_destroy_array(arr.ptr)
    register_free!(arr.nbytes)

    # update pointer & update metadata
    arr.ptr = new.ptr
    arr.nbytes = new.nbytes
    arr.padding = old_size # remember the prior (before the padding)

    # julia GC will call finalizer, but we manually cleaned it
    new.ptr = Ptr{Cvoid}(0)
    new.nbytes = 0
    new.padding = nothing
end

function add_padding(arr::NDArray, i::Int64; copy=false)
    add_padding(arr, (i,); copy=copy)
end

function check_sz!(arr, maxshape; copy=false)
    sz = cuNumeric.size(arr)
    if maxshape != nothing
        # currently require all ndarray inputs to be equal
        alligned_equal_size = sz == maxshape
        if !alligned_equal_size
            cuNumeric.add_padding(arr, maxshape; copy=copy)
            new_size = padded_shape(arr)
            @warn "[Padding Added] $sz output is now $new_size"
        end
    end
end

function check_sz(arr, maxshape)
    sz = cuNumeric.size(arr)
    if maxshape != nothing
        # currently require all ndarray inputs to be equal
        alligned_equal_size = sz == maxshape
        @assert alligned_equal_size
    end
end

function Launch(kernel::CUDATask, inputs::Tuple{Vararg{cuNumeric.NDArray}},
    outputs::Tuple{Vararg{cuNumeric.NDArray}}, scalars::Tuple{Vararg{Any}}; blocks, threads)
    input_vec = cuNumeric.VectorNDArray()

    # we find the largest input/output. Everything gets auto alligned on this.
    ndarrays = vcat(inputs..., outputs...)
    mx = findmax(arr -> arr.nbytes, ndarrays) # returns (nbytes, position)
    max_size = mx[1] # first elem nbytes
    max_shape = size(ndarrays[mx[2]]) # second elem max position
    @assert !isnothing(max_shape)

    for arr in inputs
        check_sz!(arr, max_shape; copy=true)
        cuNumeric.push_back(input_vec, CxxRef{cuNumeric.CN_NDArray}(arr.ptr))
    end

    output_vec = cuNumeric.VectorNDArray()
    for arr in outputs
        check_sz!(arr, max_shape; copy=false)
        cuNumeric.push_back(output_vec, CxxRef{cuNumeric.CN_NDArray}(arr.ptr))
    end

    scalar_vec = Legate.VectorScalar()
    for s in scalars
        Legate.push_back(scalar_vec, Legate.Scalar(s))
    end

    cuNumeric.new_task(
        kernel.func, to_stdvec(UInt32, blocks), to_stdvec(UInt32, threads), input_vec, output_vec,
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
macro cuda_task(call_expr)
    fname = call_expr.args[1]
    fargs = call_expr.args[2:end]

    esc(quote
        local _buf = IOBuffer()
        local _types = $cuNumeric.map_ndarray_cuda_types($(fargs...))
        # generate ptx using CUDA.jl 
        CUDA.code_ptx(_buf, $fname, _types; raw=false, kernel=true)

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
