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


#* NEED TO FIGURE OUT HOW TO ALLOCATE undef NDArray
Base.similar(::Type{NDArray{T,N}}, axes) where {T, N} = NDArray(similar(Array{T, N}, axes))
Base.similar(bc::Broadcasted{NDArrayStyle{N}}, ::Type{ElType}) where {N, ElType} = similar(NDArray{ElType, N}, axes(bc))

@inline function Broadcast.copy(bc::Broadcasted{<:NDArrayStyle{0}})
    error("DOES THIS NEED TO BE IMPLEMENTED? SPECIAL CASE WHEN RESULT IS SCALAR")
end

# Used to allocate output for custom implementation of broadcast
@inline function Base.copy(bc::Broadcasted{NDArrayStyle{N}}) where N
    ElType = Broadcast.combine_eltypes(bc.f, bc.args)
    if ElType == Union{} || !Base.allocatedinline(ElType)
        # a Union{} or non-isbits eltype would fail early, during GPU array construction,
        # so use a special marker to give the error a chance to be thrown during compilation
        # or even dynamically, and pick that marker up afterwards to throw an error.
        error("Cannot broadcast over types: $(eltype.(bc.args))")
    end
    copyto!(similar(bc, ElType), bc)
end

# Applies fused broadcast function to NDArray
# The operations will execute one-at-a-time for now even
# though the broadcast has been reduced to a single function.
@inline function _lower(bc::Broadcasted{<:NDArrayStyle})
    bc = Base.Broadcast.instantiate(bc)
    #* This is the place to call the CUDA.jl kernel
    #* to get actual fusion
    return bc.f(bc.args...) 
end

# Used to allocate for in-place broadcasts, for NDArray this isn't actually
# a thing as all operations on NDArrays allocate new NDArrays 
Base.copyto!(dest, bc::Broadcasted{NDArrayStyle{N}}) where N = _copyto!(dest, bc)

# Used to allocate for in-place broadcasts, for NDArray this isn't actually
# a thing as all operations on NDArrays allocate new NDArrays 
Base.copyto!(dest::NDArrayStyle{N}, bc::Broadcasted{Nothing}) where N = _copyto!(dest, bc)

function _copyto!(dest::NDArray, bc::Broadcasted)
    axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
    # isempty(dest) && return dest
    src = _lower(bc)
    copyto!(dest, src)
    return dest
end