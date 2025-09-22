using Base.Broadcast: DefaultArrayStyle, Broadcasted, AbstractArrayStyle

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

function __broadcast(f::Function, _, args...)
    error(
        """
        Tried to broadcast $(f). cuNumeric.jl does not support broadcasting user-defined functions yet. Please re-define \
        functions to match supported patterns. For example g(x) = x + 1 could be re-defined as \
        broadcast_g(x::NDArray) = x .+ 1. This can make the intention of code opaque to the reader, \
        but it is necessary until support is added."""
    )
end

# Get depth of Broadcast tree recursively 
# Need to call instantiate first 
bcast_depth(bc::Base.Broadcast.Broadcasted) = maximum(bcast_depth, bc.args, init=0) + 1;
bcast_depth(::Any) = 0

function Base.Broadcast.materialize(bc::Broadcasted{<:NDArrayStyle})
    ElType = Broadcast.combine_eltypes(bc.f, bc.args)
    if ElType == Union{} || !Base.allocatedinline(ElType)
        error("Cannot broadcast $(bc.f) over eltypes: $(eltype.(bc.args))")
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
function Base.copyto!(dest::NDArray{T, N}, bc::Broadcasted{<:NDArrayStyle{N}}) where {T,N}
    # Moves result from broadcast (src) to dest. src array is no longer valid
    #! THIS ENABLES FOOT GUN IF USER SPECIFIES INTEGER ARRAY AT OUTPUT
    nda_move(dest, checked_promote_arr(Base.Broadcast.materialize(bc), T))
    return dest
end

