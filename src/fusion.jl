export @fuse


const ALLOWED_FUSION_TYPES = Union{<:NDArray, <:Number}
const FUSED_KERNEL_CACHE = Dict{UInt64, CUDATask}()
const FUSE_KWARGS = [:blocks, :threads, :outputs]


struct FusedKernelData{I,O,S}
    ct::CUDATask
    inputs::I
    outputs::O
    scalars::S
end


#! ASSUME OUTPUT_INDICIES ARE THE ARRAYS
#! RETURNED FROM THE FUNCTION AS WELL??
#! HOW TO HANDLE CASE WHEN output_indicies = ()
#! HOW TO HANDLE KWARGS??
function fuse_function(
        fn::Function,
        output_indices,
        args...;
        kwargs...
    )
    
    println("Compiling $(Symbol(fn)) to PTX")

    N_args = length(args)
    if length(output_indices) > N_args || max(output_indices) > N_args
        error(ArgumentError("Marked arguments $(output_indices) as outputs, but there are only $(N_args) arguments"))
    end

    input_indices = setdiff(1:length(args), Set(output_indices))
    inputs = [args[i] for i in input_indices]
    outputs = [args[i] for i in output_indices]

    #! ADD KWARGS TO THESE ??
    scalars = filter(x -> isa(x, Number), inputs)
    ndarray_inputs = filter(x -> !isa(x, NDArray), inputs)
    other_inputs = filter(x -> !isa(x, Number) && !isa(x, NDArray), inputs)

    #! Parse out the isbtis structs and unwrap them??
    
    if any(isa.(outputs, Number))
        @error "Scalar outputs are not allowed, use 0D store."
    end

    # Dummy types so we can use CUDA.jl to compile
    converted_types = ndarray_cuda_type.(args) # enforces everything is an `isbitstype`
    ct = CUDATask(fn, converted_types)

    return FusedKernelData(ct, other_inputs, outputs, scalars)
end
 
# For @fuse on user defined function
function maybe_fuse_kernel(fn::Function, output_indices, args::T; kwargs...) where T

    cache_key = hash((fn, T)) #! ADD KWRAGS TO THIS?

    if !haskey(FUSED_KERNEL_CACHE, cache_key)
        FUSED_KERNEL_CACHE[cache_key] = fuse_function(
            fn,
            output_indices,
            args...;
            kwargs...
        )
    end
    
    return cache_key
end

function run_fused_kernel(cache_key; blocks, threads)    
    fkd = FUSED_KERNEL_CACHE[cache_key]
    cuNumeric.launch(fkd.ct, fkd.inputs, fkd.outputs, fkd.scalars; blocks=blocks, threads=threads)
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

    #! TODO SET DEFAULT BLOCKS/THREADS
    #! HOW TO KNOW WHATS THE RIGHT DIMENSION??

    blocks = get(kwargs[:blocks], DEFAULT_BLOCKS)
    threads = get(kwargs[:threads], DEFAULT_THREADS)
    output_indices = get(kwargs[:output_indices], ())

    code = quote end

    if Meta.isexpr(call, :function)
        # Parses function epxression into args, name etc.
        data = splitdef(longdef(call))
        # splitarg parses args into (name, type, is_slurp, default)
        arg_data = map(splitarg, data[:args])
        kwarg_data = map(splitarg, data[:kwargs])

        arg_names = getindex.(arg_data, 1)


        @gensym wrapper_name, cache_key

        push!(code.args,
            quote

                # The fused function
                function $(wrapper_name)() where {$(dict[:whereparams]...)}
                    #! NOT SURE THIS IS RIGHT WAY TO INTERPOLATE ARGS
                     #! WANT IT INTERPOLATE AS TUPLE OF ARGS
                     #! TODO PASS KWARGS
                    $cache_key = maybe_fuse_kernel($(data[:name]), $output_indices, $(arg_names...))
                    run_fused_kernel($cache_key; blocks=$blocks, threads=$threads)   
                end

                # Replace original function with call to fused
                function $(data[:name])($(data[:args]...); $(data[:kwargs]...)) where {$(dict[:whereparams]...)}
                    $(wrapper_name)()
                end
            end
        )        
    else
        throw(ArgumentError("fuse expected Broadcasted object or function, got $(call.head)"))
    end

    return esc(quote
        let
            $code
        end
    end)
end



# Collects operators and function calls from an expression.
# # E.g. x .+ y .* z will return (.+, .*)
# function collect_ops(expr::Expr)
#     ops = Symbol[] 

#     postwalk(expr) do node
#         if node isa Expr && (node.head === :call || node.head === :.)
#             op = node.args[1] 
#             if op isa Symbol
#                 push!(ops, op)
#                 # strip one leading '.' so :.+ → :+, but keep names like :sqrt
#                 # op_sym = startswith(op, ".") ? Symbol(string(op)[2:end]) : op
#                 # push!(ops, op_sym)
#             end
#         end
#         return node
#     end

#     return Tuple(ops)
# end

# This version requires all args are the same type
# @generated function maybe_fuse_kernel(
#                      :: NTuple{NARGS, NDArray{T,DIM}},
#         rhs_ops      :: NTuple{NOPS, Symbol},
#         wrapper      :: Function          
#     ) where {NARGS,T, DIM, NOPS}

#     # converted_types is constructed purely from the types of the arguments
#     # so a generated function will allow us to pre-compute most of this function
#     # at compile time!
#     converted_types = ntuple(_ -> CuDeviceArray{T, DIM, 1}, Val(NARGS))
#     _hash = xor(hash(rhs_ops), hash(converted_types))
    
#     quote
#         if !haskey($FUSED_KERNEL_CACHE, $_hash)
#             println("Compiling kernel to PTX")
#             local ptx = ptx_as_string(wrapper, $converted_types) #! DO NOT interpolate wrapper
#             f = open("/pool/emeitz/ptx2.txt", "w")
#             println(f, ptx)
#             close(f)
#             local ptx_f_name = cuNumeric.extract_kernel_name(ptx)
#             cuNumeric.ptx_task(ptx, ptx_f_name)
#             $FUSED_KERNEL_CACHE[$_hash] = FusedKernelData(ptx_f_name, $converted_types)
#         end
#         $_hash
#     end
# end

# # This version does not assume args have same types
# # Manually enforces that all args are NDArray
# @generated function maybe_fuse_kernel(rhs_args   :: R,
#                                       rhs_ops    :: NTuple{NOPS,Symbol},
#                                       wrapper    :: Function) where
#                                      {R<:Tuple, NOPS}

#     arg_types = R.parameters # e.g. (NDArray{Float32,2}, NDArray{Int,1}, …)
#     println("IN HERE")

#     for T in arg_types
#         T <: NDArray || throw(ArgumentError("all arguments must be NDArray, got $T"))
#     end

#     # 2. Build the matching CuDeviceArray type for each element -------------
#     conv_type_exprs = Expr[]
#     for T in arg_types
#         elty, dim = T.parameters[1:2] # the {T, N} in NDArray{T,N}
#         push!(conv_type_exprs, :(CuDeviceArray{$elty, $dim, 1}))
#     end
#     # an Expr(:tuple, …) is a compile‑time literal of the tuple of types
#     converted_types_expr = Expr(:tuple, conv_type_exprs...)

#     const_hash = xor(hash(rhs_ops), hash(arg_types))

#     quote
#         _hash = $const_hash                    # value is a literal, not recomputed
#         if !haskey($FUSED_KERNEL_CACHE, _hash)
#             println("Compiling kernel to PTX")
#             local ptx        = ptx_as_string(wrapper, $converted_types_expr)
#             local ptx_f_name = cuNumeric.extract_kernel_name(ptx)
#             cuNumeric.ptx_task(ptx, ptx_f_name) #! UNCOMMENT LATER
#             $FUSED_KERNEL_CACHE[_hash] =
#                 FusedKernelData(ptx_f_name, $converted_types_expr)
#         end
#         _hash
#     end
# end


# macro fuse(ex...)

#     call = ex[end]
#     kwargs = map(ex[1:end-1]) do kwarg
#         if kwarg in FUSE_KWARGS
#             :($kwarg = $kwarg)
#         else
#             throw(ArgumentError("Invalid keyword argument '$kwarg', expected one of $(FUSE_KWARGS)"))
#         end
#     end

#     code = quote end

#     if Meta.isexpr(call, :function)
#         #TODO use symbol_state.func_defs to fuse each definition
#         throw(ArgumentError("@fuse before function definitions is not supported yet."))
#     elseif Meta.isexpr(call, :(=)) || Meta.isexpr(call, :(.=)) 
#         lhs, rhs = call.args

#         lhs_symbol_state = compute_symbols_state(lhs)
#         rhs_symbol_state = compute_symbols_state(rhs)

#         if !isempty(intersect(lhs_symbol_state.references, rhs_symbol_state.references))
#             throw(ArgumentError("LHS and RHS of @fuse must not share variables."))
#         end

#         rhs_ops = collect_ops(rhs)

#         @gensym wrapper_name ptx ptx_f_name task _hash converted_types args_tuple

#         push!(code.args,
#             quote

#                 # Create wrapper function that we can fuse
#                 function $(wrapper_name)($(rhs_symbol_state.references...))
#                     # println("Wrapper called!")
#                     return $(rhs)
#                 end

#                 $_hash = $maybe_fuse_kernel(($(rhs_symbol_state.references...),), $rhs_ops, $wrapper_name)

#                 println($FUSED_KERNEL_CACHE[$_hash])

#                 $task = $(cuNumeric.CUDATask)($FUSED_KERNEL_CACHE[$_hash])
                
#                 $(cuNumeric.launch)(
#                     $task, ($(rhs_symbol_state.references...),), $lhs, (); $(kwargs...)
#                 )
#             end)
#     else
#         throw(ArgumentError("fuse expected assignment operator `=` or `.=`, got $(call.head)"))
#     end

#     return esc(quote
#         let
#             $code
#         end
#     end)
# end


#* BE SURE TO TEST:
# - x .+ y .*z and x .* y .+ z
# x,y = ...
# y = to_fuse(x)
# z = unary.(x) .+ y
# z = binary.(x,y) .+ z
# macro fuse(ex...)

#     call = ex[end]
#     kwargs = map(ex[1:end-1]) do kwarg
#         if kwarg in FUSE_KWARGS
#             :($kwarg = $kwarg)
#         else
#             throw(ArgumentError("Invalid keyword argument '$kwarg', expected one of $(FUSE_KWARGS)"))
#         end
#     end

#     code = quote end

#     if Meta.isexpr(call, :function)
#         #TODO use symbol_state.func_defs to fuse each definition
#         throw(ArgumentError("@fuse before function definitions is not supported yet."))
#     elseif Meta.isexpr(call, :(=)) || Meta.isexpr(call, :(.=)) 
#         lhs, rhs = call.args

#         lhs_symbol_state = compute_symbols_state(lhs)
#         rhs_symbol_state = compute_symbols_state(rhs)

#         if !isempty(intersect(lhs_symbol_state.references, rhs_symbol_state.references))
#             throw(ArgumentError("LHS and RHS of @fuse must not share variables."))
#         end

#         rhs_ops = collect_ops(rhs)

#         # This will error if all variables are not the same type
#         # E.g. NDArrays with different dimensions or el-types
#         rhs_args = ntuple(i -> rhs_symbol_state.references[i], length(rhs_symbol_state.references))

#         @gensym wrapper_name ptx ptx_f_name task _hash converted_types args_tuple

#         push!(code.args,
#             quote

#                 # Create wrapper function that we can fuse
#                 # $(wrapper_name) = ($(rhs_symbol_state.references...)) -> $(rhs)

#                 #* IF @fuse is in a loop this will be called multiple times for no reason...
#                 function $(wrapper_name)($(rhs_symbol_state.references...))
#                     return $(rhs)
#                 end

#                 #* IF @fuse is in a loop this will be called multiple times for no reason...
#                 $converted_types = $to_cuda_type.(($(rhs_symbol_state.references...),))
#                 $_hash = xor($hash($rhs_ops), $hash($converted_types))
                
#                 if haskey($FUSED_KERNEL_CACHE, $_hash)
#                     println("Re-using fused kernel from cache")
#                 else
#                     println("Compiling kernel to PTX")
#                     $ptx = $ptx_as_string($wrapper_name, $converted_types)
#                     $ptx_f_name = $(cuNumeric.extract_kernel_name)($ptx)
#                     $(cuNumeric.ptx_task)($ptx, $ptx_f_name)
#                     $FUSED_KERNEL_CACHE[$_hash] = $FusedKernelData($ptx_f_name, string($wrapper_name), $converted_types)
#                 end

#                 println("Args: $(($(rhs_symbol_state.references...),)), Types: $($converted_types)")
#                 println($FUSED_KERNEL_CACHE[$_hash])

#                 $task = $(cuNumeric.CUDATask)($FUSED_KERNEL_CACHE[$_hash])
                
#                 $(cuNumeric.launch)(
#                     $task, ($(rhs_symbol_state.references...),), $lhs, (); $(kwargs...)
#                 )
#             end)
#     else
#         throw(ArgumentError("fuse expected assignment operator `=` or `.=`, got $(call.head)"))
#     end

#     return esc(quote
#         let
#             $code
#         end
#     end)
# end

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
