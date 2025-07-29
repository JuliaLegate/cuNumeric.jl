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

# Collects operators and function calls from an expression.
# E.g. x .+ y .* z will return (.+, .*)
function collect_ops(expr::Expr)
    ops = Symbol[] 

    postwalk(expr) do node
        if node isa Expr && (node.head === :call || node.head === :.)
            op = node.args[1] 
            if op isa Symbol
                push!(ops, op)
                # strip one leading '.' so :.+ → :+, but keep names like :sqrt
                # op_sym = startswith(op, ".") ? Symbol(string(op)[2:end]) : op
                # push!(ops, op_sym)
            end
        end
        return node
    end

    return Tuple(ops)
end

macro fuse(ex...)

    call = ex[end]
    kwargs = map(ex[1:end-1]) do kwarg
        if kwarg in FUSE_KWARGS
            :($kwarg = $kwarg)
        else
            throw(ArgumentError("Invalid keyword argument '$kwarg', expected one of $(FUSE_KWARGS)"))
        end
    end

    code = quote end

    if Meta.isexpr(call, :function)
        #TODO use symbol_state.func_defs to fuse each definition
        throw(ArgumentError("@fuse before function definitions is not supported yet."))
    #! Right now this seems to catch tuple assignments, which we do not support (x,y = ...)
    elseif Meta.isexpr(call, :(=)) || Meta.isexpr(call, :(.=)) 
        lhs, rhs = call.args

        lhs_symbol_state = compute_symbols_state(lhs)
        rhs_symbol_state = compute_symbols_state(rhs)

        if !isempty(intersect(lhs_symbol_state.references, rhs_symbol_state.references))
            throw(ArgumentError("LHS and RHS of @fuse must not share variables."))
        end

        rhs_ops = collect_ops(rhs)

        @gensym wrapper_name ptx ptx_f_name task _hash converted_types args_tuple

        push!(code.args,
            quote

                # Create wrapper function that we can fuse
                $(wrapper_name) = ($(rhs_symbol_state.references...)) -> $(rhs)

                $args_tuple = ($(rhs_symbol_state.references...),)
                $converted_types = to_cuda_type.($args_tuple)
                $_hash = xor(hash($rhs_ops), hash($converted_types))
                
                if haskey(FUSED_KERNEL_CACHE, $_hash)
                    println("Re-using fused kernel from cache")
                else
                    println("Compiling kernel to PTX")
                    FUSED_KERNEL_CACHE[$_hash] = ptx_as_string($wrapper_name, $converted_types)
                end

                $ptx_f_name = cuNumeric.extract_kernel_name(FUSED_KERNEL_CACHE[$_hash])
                cuNumeric.ptx_task(FUSED_KERNEL_CACHE[$_hash], $ptx_f_name)
                $task = cuNumeric.CUDATask($ptx_f_name, $converted_types)
                println((($lhs)...,))
                println($lhs)
                cuNumeric.launch(
                    $task, $args_tuple, $lhs, (); $(kwargs...)
                )
            end)
    else
        throw(ArgumentError("fuse expected assignment operator `=` or `.=`, got $(call.head)"))
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
