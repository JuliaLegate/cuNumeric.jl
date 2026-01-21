
# ndarray_cuda_type(::NDArray{T,2}) where {T} = CuDeviceMatrix{T,1}
# ndarray_cuda_type(::NDArray{T,N}) where {T,N} = CuDeviceArray{T,N,1}

function ndarray_cuda_type(A::NDArray{T,N}) where {T,N}
    if N == 1
        CuDeviceVector{T,1}
    elseif N == 2
        CuDeviceMatrix{T,1}
    else
        CuDeviceArray{T,N,1}
    end
end

function ndarray_cuda_type(arg::T) where {T}
    Base.isbits(arg) || throw(ArgumentError("Unsupported argument type: $(typeof(arg))"))
    typeof(arg)
end

map_ndarray_cuda_types(args...) = tuple(map(ndarray_cuda_type, args)...)

function to_stdvec(::Type{T}, vec) where {T}
    stdvec = CxxWrap.StdVector{T}()
    for x in vec
        push!(stdvec, T(x))
    end
    return stdvec
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

# allignment contrainsts are transitive.
# we can allign all the inputs and then alligns all the outputs
# then allign one input with one output
# This reduces the need for a cartesian product.
function add_default_alignment(
    task::Legate.AutoTask, inputs::Vector{Legate.Variable}, outputs::Vector{Legate.Variable}
)
    # Align all inputs to the first input
    for i in 2:length(inputs)
        Legate.add_constraint(task, Legate.align(inputs[i], inputs[1]))
    end
    # Align all outputs to the first output
    for i in 2:length(outputs)
        Legate.add_constraint(task, Legate.align(outputs[i], outputs[1]))
    end
    # Align first output with first input
    if !isempty(inputs) && !isempty(outputs)
        Legate.add_constraint(task, Legate.align(outputs[1], inputs[1]))
    end
end

function Launch(kernel::cuNumeric.CUDATask, inputs::Tuple{Vararg{NDArray}},
    outputs::Tuple{Vararg{NDArray}}, scalars::Tuple{Vararg{Any}}; blocks, threads)

    # we find the largest input/output.
    ndarrays = vcat(inputs..., outputs...)
    mx = findmax(arr -> arr.nbytes, ndarrays) # returns (nbytes, position)
    max_size = mx[1] # first elem nbytes
    max_shape = size(ndarrays[mx[2]]) # second elem max position
    @assert !isnothing(max_shape)

    rt = Legate.get_runtime()
    lib = cuNumeric.get_lib()
    taskid = cuNumeric.RUN_PTX
    task = Legate.create_auto_task(rt, lib, taskid)

    input_vars = Vector{Legate.Variable}()
    for arr in inputs
        check_sz!(arr, max_shape; copy=true)
        store = cuNumeric.get_store(arr)
        p = Legate.add_input(task, store)
        push!(input_vars, p)
    end

    output_vars = Vector{Legate.Variable}()
    for arr in outputs
        check_sz!(arr, max_shape; copy=false)
        store = cuNumeric.get_store(arr)
        p = Legate.add_output(task, store)
        push!(output_vars, p)
    end

    # next 3 lines are reserved scalars in the RUN_PTX task
    Legate.add_scalar(task, Legate.string_to_scalar(kernel.func)) # 0
    cuNumeric.add_xyz_scalars(task, to_stdvec(UInt32, blocks))  # bx,by,bz 1,2,3
    cuNumeric.add_xyz_scalars(task, to_stdvec(UInt32, threads)) # tx,ty,tz 4,5,6

    # any user defined scalars in the launch macro
    for s in scalars
        Legate.add_scalar(task, Legate.Scalar(s)) # 7+ -> ARG_OFFSET
    end

    # all inputs are alligned with all outputs
    cuNumeric.add_default_alignment(task, input_vars, output_vars)
    Legate.submit_auto_task(rt, task)
end

function launch(kernel::cuNumeric.CUDATask, inputs, outputs, scalars; blocks, threads)
    Launch(kernel,
        isa(inputs, Tuple) ? inputs : (inputs,),
        isa(outputs, Tuple) ? outputs : (outputs,),
        isa(scalars, Tuple) ? scalars : (scalars,);
        blocks=isa(blocks, Tuple) ? blocks : (blocks,),
        threads=isa(threads, Tuple) ? threads : (threads,),
    )
end

function ptx_task(ptx::String, kernel_name)
    rt = Legate.get_runtime()
    lib = cuNumeric.get_lib() # grab lib of legate app
    # this taskid is directly tied to cpp code in our setup
    taskid = cuNumeric.LOAD_PTX
    task = Legate.create_auto_task(rt, lib, taskid)
    # assign task arguments
    Legate.add_scalar(task, Legate.string_to_scalar(ptx))
    Legate.add_scalar(task, Legate.string_to_scalar(kernel_name))
    Legate.submit_auto_task(rt, task)
end

macro cuda_task(call_expr)
    fname = call_expr.args[1]
    fargs = call_expr.args[2:end]

    esc(quote
        local _buf = IOBuffer()
        local _types = $map_ndarray_cuda_types($(fargs...))
        # generate ptx using CUDA.jl
        CUDA.code_ptx(_buf, $fname, _types; raw=false, kernel=true)

        local _ptx = String(take!(_buf))
        local _func_name = extract_kernel_name(_ptx)

        # issue ptx_task within legate runtime to register cufunction ptr with cucontext
        ptx_task(_ptx, _func_name)

        # create a cuNumeric.CUDAtask that stores some info for a launch config
        cuNumeric.CUDATask(_func_name, _types)
    end)
end

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
            launch(
                $task, $inputs, $outputs, $scalars; blocks=($blocks), threads=($threads)
            )
        end,
    )
end
