using CUDA
using Random

function __extract_kernel_name(ptx)
    for line in eachline(IOBuffer(ptx))
        if occursin(".entry", line)
            m = match(r"\.entry\s+([a-zA-Z0-9_]+)", line)
            if m !== nothing
                return String(m.captures[1])
            end
        end
    end
    return nothing
end

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

struct CUDATask
    func::Legate.LogicalStore
    argtypes::NTuple{N,Type} where {N}
end

macro cuda_task(call_expr)
    fname = call_expr.args[1]
    fargs = call_expr.args[2:end]

    esc(quote
        local _buf = IOBuffer()
        local _dummy = $cuNumeric.__dummy_args_for_ptx($(fargs...))
        # Create the PTX in runtime with actual values
        println(_dummy)
        CUDA.@device_code_ptx io=_buf CUDA.@cuda launch=false $fname((_dummy...))

        local _ptx = String(take!(_buf))
        local _func = cuNumeric.ptx_task(_ptx)
        local _types = cuNumeric.__get_types_from_dummy(_dummy)

        cuNumeric.CUDATask(_func, _types)
    end)
end

macro launch(ex...)
    @assert length(ex) == 4 "Usage: @launch task=taskname blocks=blocks threads=threads kernel(args...)"

    task = ex[1].args[2]
    blocks = ex[2].args[2]
    threads = ex[3].args[2]
    call = ex[end]

    funcname = call.args[1]    # kernel_add
    kernel_args = call.args[2:end]

    esc(
        quote
            local _task = $task

            CUDA.cudacall(
                _task.func,
                cuNumeric.__tuple_set(CUDA.KernelState, _task.argtypes...),
                _kernel_state,
                $(kernel_args...);
                threads=threads,
                blocks=blocks,
            )
            CUDA.synchronize()
        end,
    )
end
