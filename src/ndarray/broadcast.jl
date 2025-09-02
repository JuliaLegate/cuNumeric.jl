using Base.Broadcast: DefaultArrayStyle, Broadcasted, AbstractArrayStyle


#TODO Implement Broadcasting with scalars


struct NDArrayStyle{N} <: AbstractArrayStyle{N} end
Base.BroadcastStyle(::Type{<:NDArray{<:Any, N}}) where N = NDArrayStyle{N}()
Base.BroadcastStyle(::NDArrayStyle{N}, ::NDArrayStyle{M}) where {N,M} = NDArrayStyle{max(N,M)}()

_nd_forbid_mix() = throw(ArgumentError(
    "Broadcast between NDArray and other array types is not supported. " *
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


smaller_type(::Type{A}, ::Type{B}) where {A,B} = ifelse(sizeof(A) < sizeof(B), A, B)
is_wider_type(::Type{A}, ::Type{B}) where {A,B} = sizeof(A) > sizeof(B)

checked_promote_arr(arr::NDArray{T}, ::Type{T}) where T = arr

function checked_promote_arr(arr::NDArray{T}, ::Type{S}) where {T, S}
    is_wider_type(S, T) && assertpromotion(promote_type, T, S)
    return as_type(arr, S)
end

unchecked_promote_arr(arr::NDArray{T}, ::Type{T}) where T = arr
unchecked_promote_arr(arr::NDArray{T}, ::Type{S}) where {T,S} = as_type(arr, S)

# kinda hacky, but lets us support weird cases like broadcasting literal_pow
unchecked_promote_arr(::Base.RefValue{typeof(^)}, ::Type{T}) where T = typeof(Base.:(^))
unchecked_promote_arr(::Base.RefValue{Val{V}}, ::Type{T}) where {T, V} = Val{V}

__checked_promote_op(op, ::Type{Tuple{A}}) where A = __checked_promote_op(op, A)
__checked_promote_op(op, ::Type{Tuple{A, B}}) where {A,B} = __checked_promote_op(op, A, B)

# Special case for NDArray to literal integer powers
# Julia treats these special to optimize things like x ^ 2 and x ^ -1
@inline function __checked_promote_op(op::typeof(Base.literal_pow), ::Type{Tuple{_, ARR_TYPE, Val{POWER}}}) where {_, ARR_TYPE, POWER} 
    return __checked_promote_op(Base.:(^), ARR_TYPE, typeof(POWER))
end

@inline function __checked_promote_op(op, ::Type{A}) where A
    T = Base.promote_op(op, A)
    is_wider_type(T, A) && assertpromotion(op, A, T)
    return T
end

@inline function __checked_promote_op(op, ::Type{A}, ::Type{A}) where A
    T = Base.promote_op(op, A, A)
    is_wider_type(T, A) && assertpromotion(op, A, T)
    return T
end

@inline function __checked_promote_op(op, ::Type{A}, ::Type{B}) where {A, B}
    T = Base.promote_op(op, A, B)
    S = smaller_type(A, B)
    is_wider_type(T, S) && assertpromotion(op, S, T)
    return T
end

__my_promote_type(::Type{A}) where A = A
__my_promote_type(::Type{A}, ::Type{A}) where A = A

# For literal powers which are often Int64, do not check for promotion to double
# The result of promote_op with a literal integer power is always the base type
# Base.promote_op(^, Float32, Int64) == Float32
# Base.promote_op(^, Int32, Int64) == Int32
__my_promote_type(::Type{typeof(^)}, ::Type{A}, ::Type{Val{V}}) where {A, V} = promote_type(A, typeof(V))

@inline function __my_promote_type(::Type{A}, ::Type{B}) where {A,B}
    T = promote_type(A, B)
    S = smaller_type(A, B)
    is_wider_type(T, S) && assertpromotion(promote_type, S, T)
    return T
end


function __broadcast(f::Function, _, args...)
    error(
        """
        Tried to broadcast $(f). cuNumeric.jl does not support broadcasting user-defined functions yet. Please re-define \
        functions to match supported patterns. For example g(x) = x + 1 could be re-defined as \
        broadcast_g(x::NDArray) = x .+ 1. This can make the intention of code opaque to the reader, \
        but it is necessary until support is added."""
    )
end

function Base.Broadcast.materialize(bc::Broadcasted{<:NDArrayStyle})
    ElType = Broadcast.combine_eltypes(bc.f, bc.args)
    if ElType == Union{} || !Base.allocatedinline(ElType)
        error("Cannot broadcast over types: $(eltype.(bc.args))")
    end

    #* This be the place to inject kernel fusion via CUDA.jl
    #* Use the function in Base.Broadcast.flatten(bc).
    #* How can we check all the funcs in this expr 
    #* are supported by CUDA?

    return unravel_broadcast_tree(bc)
end

# Recursion base cases
__materialize(x::NDArray) = x
__materialize(x::Number) = NDArray(x)

# These are necessary to handle integer powers
__materialize(x::Base.RefValue{typeof(^)}) = x
__materialize(x::Base.RefValue{Val{-1}}) = x # enables specialized reciprocal definition
__materialize(x::Base.RefValue{Val{2}}) = x # enables specialized square definition
__materialize(x::Base.RefValue{Val{V}}) where V = NDArray(V) # Use binary_op POWER for other literal powers

# Catch unknown things...
__materialize(x) = error("Unrecognized leaf in broadcast expression: $(x)")

function __materialize(bc::Broadcasted{<:NDArrayStyle})
    bc = Base.Broadcast.instantiate(bc)
    unravel_broadcast_tree(bc)
end


function unravel_broadcast_tree(bc::Broadcasted)
    
    # Recursively materialize/unravel any nested broadcasts
    # until we reach a Broadcasted expression with only
    # NDArray or scalar arguments.
    # This is the OPPOSITE of kernel fusion 
    materialized_args = __materialize.(bc.args)

    # Handle type promotion
    eltypes = Base.Broadcast.eltypes(bc.args) 
    T_OUT = __checked_promote_op(bc.f, eltypes) # type of output array
    T_IN = __my_promote_type(eltypes.parameters...) # type input arrays are promoted to
    in_args = unchecked_promote_arr.(materialized_args, T_IN)

    # Allocate output array of proper size/type
    out = similar(NDArray{T_OUT}, axes(bc))

    # If the operation, "bc.f",  is supported by cuNumeric, this
    # dispatches to a function calling the C-API. 
    # If not it falls back to a pass-through that just calls
    # the Julia function and assumes the user defined a function
    # composed of supported operations. 
    return __broadcast(bc.f, out, in_args...)
end

# Support .= 
function Base.copyto!(dest::NDArray, bc::Broadcasted{<:NDArrayStyle})
    #! Any way to avoid the extra NDArray allocation? 
    #! materialize also allocates an output array.

    # CALL MOVE ASSIGNMENT ONTO SELF
    return copyto!(dest, Base.Broadcast.materialize(bc))
end
