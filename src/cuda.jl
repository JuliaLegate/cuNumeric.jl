using CUDA
using Random

function __get_types_from_dummy(args...)
    types = Any[]
    for arg in args
        push!(types, typeof(CUDA.cudaconvert(arg)))
    end
    return tuple(types...)
end

function __dummy_args_for_ptx(args...)
    converted = Any[]
    for arg in args
        push!(converted, cuNumeric.__convert_arg(arg))
    end
    return tuple(converted...)
end

function __convert_arg(arg)
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

function __tuple_set(args...)
    state = args[1]
    types = args[2]

    t = Any[]
    push!(t, state)
    for ty in types.parameters
        push!(t, ty)
    end
    return tuple(t...)
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
        local _dummy = $cuNumeric.__dummy_args_for_ptx($(fargs...))
        # Create the PTX in runtime with actual values
        # old PTX generation
        # CUDA.@device_code_ptx io=_buf CUDA.@cuda launch=false $fname((_dummy...))
        # Tim reccomends the following:

        CUDA.code_ptx

        local _ptx = String(take!(a_buf))
        local _func_name = cuNumeric.extract_kernel_name(_ptx)
        local _func = cuNumeric.ptx_task(_ptx, _func_name)
        local _types = cuNumeric.__get_types_from_dummy(_dummy)

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
