const FUSED_KERNEL_CACHE = Dict{UInt64, String}() # xor(hash(f_name), hash(list of types)) => PTX
const FUSE_KWARGS = [:blocks, :threads]

# For testing purpsoes
# struct NDArray{T, N} end
# Base.eltype(::NDArray{T}) where T = T
# Base.ndims(::NDArray{T, N}) where {T, N} = N

function ptx_as_string(f::Function, types::Tuple)
    buf = IOBuffer()
    CUDA.code_ptx(buf, f, types; raw = true)
    return String(take!(buf))
end

# function extract_kernel_name(ptx::String)
#     # Regex with a named capture “name”
#     re = r"\.visible\s+\.entry\s+(?<name>[_A-Za-z0-9\$]+)"
#     m = match(re, ptx)
#     if m === nothing
#         error("Could not find a `.visible .entry <name>` directive in PTX")
#     end
#     return m.captures[1]
# end

# function to_cuda_type(::NDArray{T,N}) where {T,N}
#    return CuDeviceArray{T, N, 1}
# end


## Place in front of a function call (CHANGE THIS LATER TO SUPPORT ARBITRARY BROADCASTS)
## to compile/fuse with CUDA.jl and launch 
## as a CUDA kernel through Legate. 
# e.g. @fuse y = to_fuse(x, y), will break for splats and funcs with names args.
# assumes 1D as well
macro fuse(ex...)

    call = ex[end]
    kwargs = map(ex[1:end-1]) do kwarg
        if kwarg in FUSE_KWARGS
            :($kwarg = $kwarg)
        else
            throw(ArgumentError("Invalid keyword argument '$kwarg', expected one of $(FUSE_KWARGS)"))
        end
    end

    Meta.isexpr(call, :(=)) || error("fuse macro only supports assignments to unbroadcasted function calls")
    out_var, func = call.args
    Meta.isexpr(func, :call) || throw(ArgumentError("fuse macro only supports assignments to unbroadcasted function calls"))
    f_name, f_args... = func.args

    @gensym ptx ptx_f_name task _hash converted_types args_tuple

    code = quote

        $args_tuple = ($(f_args...),)
        println($args_tuple)
        $converted_types = to_cuda_type.($args_tuple)
        $_hash = xor(hash($f_name), hash($converted_types))
        
        if haskey(FUSED_KERNEL_CACHE, $_hash)
            println("Re-using fused kernel from cache")
        else
            println("Compiling kernel to PTX")
            FUSED_KERNEL_CACHE[$_hash] = ptx_as_string($f_name, $converted_types)
        end

        $ptx_f_name = cuNumeric.extract_kernel_name(FUSED_KERNEL_CACHE[$_hash])
        cuNumeric.ptx_task(FUSED_KERNEL_CACHE[$_hash], $ptx_f_name)
        $task = cuNumeric.CUDATask($ptx_f_name, $converted_types)

        cuNumeric.launch(
            $task, $args_tuple, $(out_var,), (); $(kwargs...) # requires kwrags to be defined in cuNumeric.launch
        )
    end

    return esc(quote
        let
            $code
        end
    end)
end

# function to_fuse(x)
#     T = eltype(x)
#     return exp.(x.*x) .+ T(1.0) 
# end

# x = NDArray{Float32, 1}()
# @fuse y = to_fuse(x)



# function kernel(x)
#     tid = threadIdx().x
#     if tid <= length(x)
#         x[tid] = x[tid] + 1.0f0
#     end
# end
