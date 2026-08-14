export @cuda_task, @launch, CUDATask

struct CUDATask
    func::String
    argtypes::Vector{DataType}

    function CUDATask(func, argtypes)
        return new(convert(String, func), collect(DataType, argtypes))
    end
end

#! JUST PASS TYPES HERE INSTEAD OF CALLING typeof()
map_ndarray_cuda_types(args...) = tuple(map(ndarray_cuda_type, typeof.(args))...)

function to_stdvec(::Type{T}, vec) where {T}
    stdvec = CxxWrap.StdVector{T}()
    for x in vec
        push!(stdvec, T(x))
    end
    return stdvec
end

@inline function _launch_shape(arr::NDArray)
    padding = _padding(arr)
    return isnothing(padding) ? size(arr) : padding.shape
end

function _ensure_launch_padding!(arr::NDArray{T,N}, target_shape; copy=false) where {T,N}
    padding = _padding(arr)
    !isnothing(padding) && padding.shape == target_shape && return arr
    isnothing(padding) && size(arr) == target_shape && return arr
    @assert all(target_shape .>= size(arr)) "cannot pad $(size(arr)) to $target_shape"

    padded = zeros(T, target_shape)
    slices = ntuple(d -> (0, size(arr, d)), N)
    logical = nda_get_slice(padded, slice_array(slices...))
    aliases_parent = !isnothing(arr.parent)
    copy && !aliases_parent && copyto!(logical, arr)
    storage = PaddedStorage{T,N}(
        padded,
        aliases_parent ? logical : nothing,
        target_shape,
    )

    if aliases_parent
        old_padding = _padding(arr)
        arr.padding = storage
        !isnothing(old_padding) && _destroy_padded_storage!(old_padding)
    else
        destroy!(arr)
        arr.ptr = logical.ptr
        arr.nbytes = logical.nbytes
        arr.padding = storage
        logical.ptr = Ptr{Cvoid}(0)
        logical.nbytes = 0
    end
    return arr
end

function _sync_to_launch_padding!(arr::NDArray)
    padding = _padding(arr)
    if !isnothing(padding) && !isnothing(padding.staging)
        copyto!(padding.staging, arr)
    end
    return nothing
end

function _sync_from_launch_padding!(arr::NDArray)
    padding = _padding(arr)
    if !isnothing(padding) && !isnothing(padding.staging)
        copyto!(arr, padding.staging)
    end
    return nothing
end

# `get_store` returns a Julia-owned `LogicalArrayImplAllocated` that shares the
# underlying Legate store with the NDArray. `add_input`/`add_output` copy that
# array into the task; if we leave the temporary alive until GC, store refcounts
# stay elevated and framebuffer reclaim stalls (fusion 1-GPU OOM under load).
# Finalize the temporary immediately after the copy into the task.
function _add_task_array!(add_to, task, arr::NDArray; physical=false)
    padding = _padding(arr)
    task_arr = physical && !isnothing(padding) ? padding.backing : arr
    st = cuNumeric.get_store(task_arr)
    try
        return add_to(task, st)
    finally
        finalize(st)
    end
end

function Launch(kernel::CUDATask, inputs::Tuple{Vararg{NDArray}},
    outputs::Tuple{Vararg{NDArray}}, scalars::Tuple{Vararg{Any}};
    blocks, threads, taskid=cuNumeric.RUN_PTX, ctx=nothing)
    rt = Legate.get_runtime()
    lib = cuNumeric.get_lib()
    task = Legate.create_auto_task(rt, lib, taskid)
    physical = taskid == cuNumeric.RUN_PTX

    input_vars = Vector{Legate.Variable}()
    for arr in inputs
        push!(input_vars, _add_task_array!(Legate.add_input, task, arr; physical))
    end

    output_vars = Vector{Legate.Variable}()
    for arr in outputs
        push!(output_vars, _add_task_array!(Legate.add_output, task, arr; physical))
    end

    # Reserved scalars: kernel_name (0), blocks (1,2,3), threads (4,5,6)
    Legate.add_scalar(task, Legate.string_to_scalar(kernel.func)) # 0
    cuNumeric.add_xyz_scalars(task, to_stdvec(UInt32, blocks))    # 1,2,3
    cuNumeric.add_xyz_scalars(task, to_stdvec(UInt32, threads))   # 4,5,6

    # CompilerMetadata ctx for broadcast tasks (scalar 7, raw bytes)
    if !isnothing(ctx)
        ref = Ref(ctx)
        GC.@preserve ref begin
            cuNumeric.add_scalar_from_ptr(task, Base.unsafe_convert(Ptr{Cvoid}, ref), sizeof(ctx))
        end
    end

    # User-defined scalars
    for s in scalars
        Legate.add_scalar(task, Legate.Scalar(s))
    end

    # all inputs are aligned with all outputs
    Legate.default_alignment(task, input_vars, output_vars)
    return Legate.submit_auto_task(rt, task)
end

function launch(kernel::CUDATask, inputs, outputs, scalars;
    blocks, threads, taskid=cuNumeric.RUN_PTX, ctx=nothing)
    input_tuple = isa(inputs, Tuple) ? inputs : (inputs,)
    output_tuple = isa(outputs, Tuple) ? outputs : (outputs,)

    # Custom tasks require equal physical shapes. Keep the padded backing so
    # repeated launches do not allocate or copy again.
    if taskid == cuNumeric.RUN_PTX
        arrays = (input_tuple..., output_tuple...)
        if !isempty(arrays)
            rank = ndims(first(arrays))
            @assert all(ndims(arr) == rank for arr in arrays) "custom task arrays must have equal ranks"
            max_shape = ntuple(d -> maximum(_launch_shape(arr)[d] for arr in arrays), rank)
            foreach(arr -> _ensure_launch_padding!(arr, max_shape; copy=true), input_tuple)
            foreach(arr -> _ensure_launch_padding!(arr, max_shape), output_tuple)
            foreach(_sync_to_launch_padding!, input_tuple)
        end
    end

    result = Launch(kernel,
        input_tuple,
        output_tuple,
        isa(scalars, Tuple) ? scalars : (scalars,);
        blocks=isa(blocks, Tuple) ? blocks : (blocks,),
        threads=isa(threads, Tuple) ? threads : (threads,),
        taskid=taskid, ctx=ctx,
    )
    taskid == cuNumeric.RUN_PTX && foreach(_sync_from_launch_padding!, output_tuple)
    return result
end

function ptx_task(ptx::String, kernel_name)
    rt = Legate.get_runtime()
    lib = cuNumeric.get_lib()
    taskid = cuNumeric.LOAD_PTX

    # One point task per GPU so every GPU compiles the module.
    ngpus = max(Int(Legate.num_gpus()), 1)
    domain = Legate.domain_from_shape(Legate.Shape(Legate.to_cxx_vector((ngpus,))))
    task = Legate.create_manual_task(rt, lib, taskid, domain)
    Legate.add_scalar(task, Legate.string_to_scalar(ptx))
    Legate.add_scalar(task, Legate.string_to_scalar(kernel_name))
    Legate.submit_manual_task(rt, task)

    # Fence so every load finishes before any launch reads the cache.
    return issue_execution_fence(; block=false)
end

function _emit_compatible_ptx(io, f, types)
    ptx_version = _COMPATIBLE_PTX_VERSION[]
    # dump_module=true keeps linked libdevice helpers in the emitted module.
    return CUDATools.code_ptx(
        io, f, types; raw=false, dump_module=true, kernel=true, ptx=ptx_version
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
    cuNumeric.assert_experimental()

    fname = call_expr.args[1]
    fargs = call_expr.args[2:end]

    return esc(
        quote
            local _buf = IOBuffer()
            local _types = cuNumeric.map_ndarray_cuda_types($(fargs...))
            # generate ptx using CUDA.jl
            cuNumeric._emit_compatible_ptx(_buf, $fname, _types)

            local _ptx = String(take!(_buf))
            local _func_name = cuNumeric.extract_kernel_name(_ptx)

            # issue ptx_task within legate runtime to register cufunction ptr with cucontext
            cuNumeric.ptx_task(_ptx, _func_name)

            # create a cuNumeric.CUDAtask that stores some info for a launch config
            cuNumeric.CUDATask(_func_name, _types)
        end,
    )
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
    cuNumeric.assert_experimental()

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

    return esc(
        quote
            cuNumeric.launch(
                $task, $inputs, $outputs, $scalars;
                blocks=($blocks), threads=($threads),
            )
        end,
    )
end
