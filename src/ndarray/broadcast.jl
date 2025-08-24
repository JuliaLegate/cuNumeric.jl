using Base.Broadcast: DefaultArrayStyle, Broadcasted, AbstractArrayStyle


#TODO Implement Broadcasting with scalars


struct NDArrayStyle{N} <: AbstractArrayStyle{N} end
Base.BroadcastStyle(::Type{<:NDArray{<:Any, N}}) where N = NDArrayStyle{N}()
Base.BroadcastStyle(::NDArrayStyle{N}, ::NDArrayStyle{M}) where {N,M} = NDArrayStyle{max(N,M)}()

_nd_forbid_mix() = throw(ArgumentError(
    "Broadcast between NDArray and regular arrays is not supported. " *
    "Convert explicitly to a single array type before broadcasting."
))


# Allow broadcasting with scalars
Base.BroadcastStyle(a::NDArrayStyle, ::DefaultArrayStyle{0}) = a
Base.BroadcastStyle(::DefaultArrayStyle{0}, a::NDArrayStyle) = a

# Disallow broadcasting with normal arrays
Base.BroadcastStyle(::NDArrayStyle, ::DefaultArrayStyle) = _nd_forbid_mix()
Base.BroadcastStyle(::DefaultArrayStyle, ::NDArrayStyle) = _nd_forbid_mix()

Base.broadcastable(A::NDArray) = A


#* IS THERE A BETTER WAY TO ALLOCATE THE NEW ARRAY???
Base.similar(arr::NDArray, ::Type{T}, dims::Dims{N}) where {T,N} = cuNumeric.zeros(T, dims)
Base.similar(arr::NDArray, ::Type{T}, dims::Base.DimOrInd...) where T = similar(arr, T, dims)
Base.similar(arr::NDArray{T,N}) where {T,N} = similar(arr, T, size(arr))
Base.similar(arr::NDArray{T}, dims::Tuple) where T = similar(arr, T, dims)
Base.similar(arr::NDArray{T}, dims::Base.DimOrInd...) where T = similar(arr, T, dims)
Base.similar(arr::NDArray, ::Type{T}) where T = similar(arr, T, size(arr))

#* IS THERE A BETTER WAY TO ALLOCATE THE NEW ARRAY???
Base.similar(::Type{NDArray{T}}, axes) where T = cuNumeric.zeros(T, Base.to_shape.(axes))
Base.similar(bc::Broadcasted{NDArrayStyle{N}}, ::Type{ElType}) where {N, ElType} = similar(NDArray{ElType}, axes(bc))

maybe_promote_arr(arr::NDArray{T}, ::Type{T}) where T = arr
maybe_promote_arr(arr::NDArray{T}, ::Type{S}) where {T,S} = as_type(arr, S)

smaller_type(::Type{A}, ::Type{B}) where {A,B} = ifelse(sizeof(A) < sizeof(B), A, B)
same_size(::Type{A}, ::Type{B}) where {A,B} = sizeof(A) == sizeof(B)

__my_promote_type(::Type{Tuple{A, B}}) where {A,B} = __my_promote_type(A, B)

function __my_promote_type(::Type{A}, ::Type{B}) where {A, B}
    T = promote_type(A, B)
    same_size(A, B) && return T
    S = smaller_type(A, B)
    S != T && error("Detected promotion from $S to larger type, $T")
    return T
end


function __maybe_apply(f::Function, _, args...)
    error(
        """
        cuNumeric.jl does not support broadcasting user-defined functions yet. Please re-define \
        functions to match supported patterns. For example g(x) = x + 1 could be re-defined as \
        broadcast_g(x::NDArray) = x .+ 1. This can make the intention of code opaque to the reader, \
        but it is necessary until support is added."""
    )
end

@inline function __maybe_apply(f::typeof(+), out::NDArray{T}, rhs1::NDArray{T}, rhs2::NDArray{T}) where {T <: SUPPORTED_TYPES}
    return nda_binary_op(out, cuNumeric.ADD, rhs1, rhs2)
end

@inline function __maybe_apply(f::typeof(*), out::NDArray{T}, rhs1::NDArray{T}, rhs2::NDArray{T}) where {T <: SUPPORTED_TYPES}
    return nda_binary_op(out, cuNumeric.MULTIPLY, rhs1, rhs2)
end

@inline function __maybe_apply(f::typeof(Base.sin), out::NDArray{T}, arr::NDArray{T}) where {T <: SUPPORTED_TYPES}
    return nda_unary_op(out, cuNumeric.SIN, arr)
end

function Base.Broadcast.materialize(bc::Broadcasted{<:NDArrayStyle})

    ElType = Broadcast.combine_eltypes(bc.f, bc.args)
    if ElType == Union{} || !Base.allocatedinline(ElType)
        error("Cannot broadcast over types: $(eltype.(bc.args))")
    end

    #* This be the place to inject kernel fusion via CUDA.jl
    return unravel_broadcast_tree(bc)
end

# Recursion base case
__materialize(x::NDArray) = x

function __materialize(bc::Broadcasted{<:NDArrayStyle})
    bc = Base.Broadcast.instantiate(bc)
    unravel_broadcast_tree(bc)
end

function unravel_broadcast_tree(bc::Broadcasted)
    
    # Recursively materialize/unravel any nested broadcasts
    # This is the OPPOSITE of kernel fusion 
    materialized_args = __materialize.(bc.args)

    # Handle type promotion
    eltypes = Base.Broadcast.eltypes(bc.args)
    T = __my_promote_type(eltypes)
    in_args = maybe_promote_arr.(materialized_args, T)

    # Allocate output array of proper size/type
    out = similar(NDArray{T}, axes(bc))

    # If the operation, "bc.f",  is supported by cuNumeric, this
    # dispatches to a function calling the C-API. 
    # If not it falls back to a pass-through that just calls
    # the Julia function and assumes the user defined a function
    # composed of supported operations. 
    return __maybe_apply(bc.f, out, in_args...)
end


# @inline function Broadcast.copy(bc::Broadcasted{<:NDArrayStyle{0}})
#     error("DOES THIS NEED TO BE IMPLEMENTED? SPECIAL CASE WHEN RESULT IS SCALAR")
# end

# # Used to allocate output for custom implementation of broadcast
# @inline function Base.copy(bc::Broadcasted{NDArrayStyle{N}}) where N
#     ElType = Broadcast.combine_eltypes(bc.f, bc.args)
#     if ElType == Union{} || !Base.allocatedinline(ElType)
#         # a Union{} or non-isbits eltype would fail early, during GPU array construction,
#         # so use a special marker to give the error a chance to be thrown during compilation
#         # or even dynamically, and pick that marker up afterwards to throw an error.
#         error("Cannot broadcast over types: $(eltype.(bc.args))")
#     end
#     copyto!(similar(bc, ElType), bc)
# end

# # Applies fused broadcast function to NDArray
# # The operations will execute one-at-a-time for now even
# # though the broadcast has been reduced to a single function.
# @inline function _lower(bc::Broadcasted{<:NDArrayStyle})
#     bc = Base.Broadcast.instantiate(bc)
#     #* This is the place to call the CUDA.jl kernel
#     #* to get actual fusion
#     println(bc)
#     bc = Base.Broadcast.flatten(bc) # converts all to NDArray isntead of Broadcasted{NDarray}
#     println(bc)
#     println(bc.args)
#     if haskey(binary_op_map, bc.f)
#         println("Calling internal function: $(binary_op_map[bc.f][2])")
#         eval(binary_op_map[bc.f][2])(bc.args...)
#     else
#         println("Unrecognized function: $(bc.f)")
#         return bc.f(bc.args...)
#     end
# end

# # Used to allocate for in-place broadcasts, for NDArray this isn't actually
# # a thing as all operations on NDArrays allocate new NDArrays 
# Base.copyto!(dest, bc::Broadcasted{NDArrayStyle{N}}) where N = _copyto!(dest, bc)

# # Used to allocate for in-place broadcasts, for NDArray this isn't actually
# # a thing as all operations on NDArrays allocate new NDArrays 
# Base.copyto!(dest::NDArrayStyle{N}, bc::Broadcasted{Nothing}) where N = _copyto!(dest, bc)

# function _copyto!(dest::NDArray, bc::Broadcasted)
#     axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
#     # isempty(dest) && return dest
#     src = _lower(bc)
#     copyto!(dest, src)
#     return dest
# end