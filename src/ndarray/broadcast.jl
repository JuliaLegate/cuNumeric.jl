using Base.Broadcast: DefaultArrayStyle, Broadcasted, AbstractArrayStyle

struct NDArrayStyle{N} <: AbstractArrayStyle{N} end
Base.BroadcastStyle(::Type{<:NDArray{<:Any,N}}) where {N} = NDArrayStyle{N}()
Base.BroadcastStyle(::NDArrayStyle{N}, ::NDArrayStyle{M}) where {N,M} = NDArrayStyle{max(N, M)}()

function _nd_forbid_mix()
    throw(
        ArgumentError(
            "Broadcast between NDArray and other array types is not supported. " *
            "Convert explicitly to a single array type before broadcasting.",
        ),
    )
end

# Allow broadcasting with scalars
Base.BroadcastStyle(a::NDArrayStyle, ::DefaultArrayStyle{0}) = a
Base.BroadcastStyle(::DefaultArrayStyle{0}, a::NDArrayStyle) = a

# Disallow broadcasting with normal arrays
Base.BroadcastStyle(::NDArrayStyle, ::DefaultArrayStyle) = _nd_forbid_mix()
Base.BroadcastStyle(::DefaultArrayStyle, ::NDArrayStyle) = _nd_forbid_mix()

Base.broadcastable(A::NDArray) = A

#* IS THERE A BETTER WAY TO ALLOCATE THE NEW ARRAY???
Base.similar(arr::NDArray, ::Type{T}, dims::Dims{N}) where {T,N} = cuNumeric.zeros(T, dims)
Base.similar(arr::NDArray, ::Type{T}, dims::Base.DimOrInd...) where {T} = similar(arr, T, dims)
Base.similar(arr::NDArray{T,N}) where {T,N} = similar(arr, T, size(arr))
Base.similar(arr::NDArray{T}, dims::Tuple) where {T} = similar(arr, T, dims)
Base.similar(arr::NDArray{T}, dims::Base.DimOrInd...) where {T} = similar(arr, T, dims)
Base.similar(arr::NDArray, ::Type{T}) where {T} = similar(arr, T, size(arr))

#* IS THERE A BETTER WAY TO ALLOCATE THE NEW ARRAY???
Base.similar(::Type{NDArray{T}}, axes) where {T} = cuNumeric.zeros(T, Base.to_shape.(axes))
function Base.similar(bc::Broadcasted{NDArrayStyle{N}}, ::Type{ElType}) where {N,ElType}
    similar(NDArray{ElType}, axes(bc))
end

function __broadcast(f::Function, _, args...)
    #! WITH FUSION I THINK WE CAN SUPPORT THIS BY JUST CALLING MAP or MAP!
    error(
        """
        Tried to broadcast $(f). cuNumeric.jl does not support broadcasting user-defined functions yet. Please re-define \
        functions to match supported patterns. For example g(x) = x + 1 could be re-defined as \
        broadcast_g(x::NDArray) = x .+ 1. This can make the intention of code opaque to the reader, \
        but it is necessary until support is added.""",
    )
end

# Get depth of Broadcast tree recursively
# Need to call instantiate first
bcast_depth(bc::Base.Broadcast.Broadcasted) = maximum(bcast_depth, bc.args, init=0) + 1;
bcast_depth(::Any) = 0

# Copied from GPUArrays: https://github.com/JuliaGPU/GPUArrays.jl/blob/a9df2ba41ca2358c1de2f3cc6b020578bf6e39b1/src/host/broadcast.jl#L60-L63
# Defined with KernelAbstractions.jl. Makes it easier to generate indexing for
# various dimensions of inputs/outputs. Assumes broadcast is `Base.Broadcast.process`ed so that
# dest/bc have singleton dimensions inserted and we can index 1-1 like this.
@kernel function broadcast_kernel_cartesian(dest, bc)
    I = @index(Global, Cartesian)
    @inbounds dest[I] = bc[I]
end

@kernel function broadcast_kernel_linear(dest, bc)
    I = @index(Global, Linear)
    @inbounds dest[I] = bc[I]
end

# No compilation here, just generating CUDA specific kernel.
const GPU_CARTESIAN_KERNEL = broadcast_kernel_cartesian(CUDACore.CUDAKernels.CUDABackend())
const GPU_LINEAR_KERNEL = broadcast_kernel_linear(CUDACore.CUDAKernels.CUDABackend())

struct BrokenBroadcast{T} end
Base.convert(::Type{BrokenBroadcast{T}}, x) where {T} = BrokenBroadcast{T}()
Base.convert(::Type{BrokenBroadcast{T}}, x::BrokenBroadcast{T}) where {T} = x
Base.eltype(::Type{BrokenBroadcast{T}}) where {T} = T

function Broadcast.copy(bc::Broadcasted{<:NDArrayStyle{0}})
    ElType = Broadcast.combine_eltypes(bc.f, bc.args)
    if ElType == Union{}
        ElType = Nothing
    end
    dest = copyto!(similar(bc, ElType), bc)
    #! CHECK THIS DOESNT CAUSE ISSUES DUE TO BLOCKING NATURE
    return @allowscalar dest[CartesianIndex()]
end

@inline function Broadcast.copy(bc::Broadcasted{<:NDArrayStyle})
    ElType = Broadcast.combine_eltypes(bc.f, bc.args)
    if ElType == Union{} || !Base.allocatedinline(ElType)
        ElType = BrokenBroadcast{ElType}
    end
    copyto!(similar(bc, ElType), bc)
end

# Recursion base cases
__materialize(x::NDArray) = x
__materialize(x::Number) = NDArray(x)

# These are necessary to handle integer powers
__materialize(x::Base.RefValue{typeof(^)}) = x
__materialize(x::Base.RefValue{Val{-1}}) = x # enables specialized reciprocal definition
__materialize(x::Base.RefValue{Val{2}}) = x # enables specialized square definition
__materialize(x::Base.RefValue{Val{V}}) where {V} = NDArray(V) # Use binary_op POWER for other literal powers

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

function fuse_broadcast_tree!(dest::NDArray, bc::Broadcasted)
    bc = Base.Broadcast.preprocess(dest, bc)

    # Get proper kernel
    broadcast_kernel =
        if ndims(dest) == 1 ||
            (isa(IndexStyle(dest), IndexLinear) &&
            isa(IndexStyle(bc), IndexLinear))
            GPU_LINEAR_KERNEL
        else
            GPU_CARTESIAN_KERNEL
        end

    #! DO I NEED TO DO TYPE PROMOTION CHECKS??
    # ndims check for 0D support
    broadcast_kernel(dest, bc; ndrange=ndims(dest) > 0 ? size(dest) : (1,))
    return dest
end

@inline function _copyto!(dest::NDArray, bc::Broadcasted)
    axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
    isempty(dest) && return dest
    if eltype(dest) <: BrokenBroadcast
        throw(
            ArgumentError(
                "Broadcast operation resulting in $(eltype(eltype(dest))) is not NDArray compatible"
            ),
        )
    end

    #! IF THIS IS KNOWN AT COMPILE TIME WE CAN GENERATE
    #! THIS CODE WITHOUT THE IF STATEMENT
    if FUSE_BROADCAST_EXPRS
        #! DO I NEED TO DO TYPE PROMOTION CHECKS BEFORE RETURNING?
        #! WE MIGHT NEED TO CHECK IF ON GPU OR CPU AND FALLBACK
        return fuse_broadcast_tree!(dest, bc)
    else
        temp_result = unravel_broadcast_tree(bc)
        nda_move(dest, checked_promote_arr(temp_result, eltype(dest)))
        return dest
    end
end

# Support .=
@inline Base.copyto!(dest::NDArray, bc::Broadcasted{Nothing}) = _copyto!(dest, bc)
@inline Base.copyto!(dest::NDArray, bc::Broadcasted{<:NDArrayStyle}) = _copyto!(dest, bc)

#! TODO ADD MAP FUSED IMPLEMENTATIONS
