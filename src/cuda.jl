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
        return CUDA.zeros(T, 1)
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

macro cuda_task(call_expr)
    fname = call_expr.args[1]
    fargs = call_expr.args[2:end]

    esc(quote
        local _buf = IOBuffer()
        local _types = $cuNumeric.map_ndarray_cuda_types($(fargs...))
        # generate ptx using CUDA.jl 
        CUDA.code_ptx(_buf, $fname, _types; raw=false)

        local _ptx = String(take!(_buf))
        local _func_name = cuNumeric.extract_kernel_name(_ptx)
        println(_ptx)
        println(_func_name)

        # issue ptx_task within legate runtime to register cufunction ptr with cucontext
        cuNumeric.ptx_task(_ptx, _func_name)

        # create a CUDAtask that stores some info for a launch config
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
            cuNumeric.launch(
                $task, $inputs, $outputs, $scalars; blocks=($blocks), threads=($threads)
            )
        end,
    )
end
