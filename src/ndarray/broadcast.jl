using Base.Broadcast: DefaultArrayStyle, Broadcasted, AbstractArrayStyle

struct NDArrayStyle{N} <: AbstractArrayStyle{N} end
Base.BroadcastStyle(::Type{<:NDArray{<:Any,N}}) where {N} = NDArrayStyle{N}()
Base.BroadcastStyle(::NDArrayStyle{N}, ::NDArrayStyle{M}) where {N,M} = NDArrayStyle{max(N, M)}()

# Some other functions in cuda_util.jl
function map_cuda_type(::Type{cuNumeric.NDArrayStyle{N}}) where {N}
    return CUDACore.CuArrayStyle{N,CUDACore.DeviceMemory}
end # Also can be HostMemory or UnifiedMemory

function _nd_forbid_mix()
    return throw(
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

# Like Base Diagonal vs Array: structured Diagonal style wins over dense NDArray
# so D.+A uses StructuredMatrixStyle{Diagonal} (densify to NDArray in diagonal.jl)
# instead of ArrayConflict → host Matrix + scalar indexing.
function Base.BroadcastStyle(
    ::LinearAlgebra.StructuredMatrixStyle{<:Diagonal}, ::NDArrayStyle
)
    return LinearAlgebra.StructuredMatrixStyle{Diagonal}()
end
function Base.BroadcastStyle(
    ::NDArrayStyle, ::LinearAlgebra.StructuredMatrixStyle{<:Diagonal}
)
    return LinearAlgebra.StructuredMatrixStyle{Diagonal}()
end

Base.broadcastable(A::NDArray) = A

Base.similar(arr::NDArray, ::Type{T}, dims::Dims{N}) where {T,N} = NDArray{T}(undef, dims)
Base.similar(arr::NDArray, ::Type{T}, dims::Base.DimOrInd...) where {T} = similar(arr, T, dims)
Base.similar(arr::NDArray{T,N}) where {T,N} = similar(arr, T, size(arr))
Base.similar(arr::NDArray{T}, dims::Tuple) where {T} = similar(arr, T, dims)
Base.similar(arr::NDArray{T}, dims::Base.DimOrInd...) where {T} = similar(arr, T, dims)
Base.similar(arr::NDArray, ::Type{T}) where {T} = similar(arr, T, size(arr))

# Prefer Dims over the axes catch-all: with StaticArrays loaded (GPU CI via CUDA),
# `similar(::Type{<:AbstractArray}, ::Tuple{})` is otherwise ambiguous between
# Base, StaticArrays, and our catch-all (0-d broadcast uses axes `()`).
Base.similar(::Type{NDArray{T}}, dims::Dims{N}) where {T,N} = NDArray{T}(undef, dims)
function Base.similar(
    ::Type{NDArray{T}},
    shape::Tuple{Union{Integer,Base.OneTo},Vararg{Union{Integer,Base.OneTo}}},
) where {T}
    return NDArray{T}(undef, map(Int, Base.to_shape.(shape)))
end
Base.similar(::Type{NDArray{T}}, axes) where {T} = NDArray{T}(undef, Base.to_shape.(axes))
function Base.similar(bc::Broadcasted{NDArrayStyle{N}}, ::Type{ElType}) where {N,ElType}
    return similar(NDArray{ElType}, axes(bc))
end

function __broadcast(f::Function, _, args...)
    return error(
        "Broadcasting $(f) is not supported by cuNumeric's unfused broadcast path. " *
        "Functions without a native broadcast implementation require GPU fusion, compatible array shapes, " *
        "and a GPU-compilable function (fusion enabled: $(FUSE_BROADCAST_EXPRS)). " *
        "Otherwise, rewrite the expression using supported broadcast operations.",
    )
end

# Low-storage Runge–Kutta methods use muladd. Express it as two supported
# broadcasts so the unfused path works and GPU fusion can still combine them.
@inline function __broadcast(
    ::typeof(muladd), out::NDArray, a::NDArray, b::NDArray, c::NDArray
)
    out .= a .* b .+ c
    return out
end

# Get depth of Broadcast tree recursively
# Need to call instantiate first
bcast_depth(bc::Base.Broadcast.Broadcasted) = maximum(bcast_depth, bc.args; init=0) + 1;
bcast_depth(::Any) = 0

struct BrokenBroadcast{T} end
Base.convert(::Type{BrokenBroadcast{T}}, x) where {T} = BrokenBroadcast{T}()
Base.convert(::Type{BrokenBroadcast{T}}, x::BrokenBroadcast{T}) where {T} = x
Base.eltype(::Type{BrokenBroadcast{T}}) where {T} = T

# Use cuNumeric promotion (`__recip_type` for inv, etc.), not Base.combine_eltypes
# — e.g. inv.(Int32) must allocate Float32, not Float64.
@inline function _broadcast_copy_eltype(bc::Broadcasted)
    return __checked_promote_op(bc.f, Base.Broadcast.eltypes(bc.args))
end

function Broadcast.copy(bc::Broadcasted{<:NDArrayStyle{0}})
    ElType = _broadcast_copy_eltype(bc)
    if ElType == Union{}
        ElType = Nothing
    end
    return copyto!(similar(bc, ElType), bc)
end

@inline function Broadcast.copy(bc::Broadcasted{<:NDArrayStyle})
    ElType = _broadcast_copy_eltype(bc)
    if ElType == Union{} || !Base.allocatedinline(ElType)
        ElType = BrokenBroadcast{ElType}
    end
    return copyto!(similar(bc, ElType), bc)
end

# Recursion base cases
__materialize(x::NDArray) = x
# Keep Numbers as scalars; unchecked_promote_arr builds the 0-d NDArray once.
__materialize(x::Number) = x

# These are necessary to handle integer powers
__materialize(x::Base.RefValue{typeof(^)}) = x
__materialize(x::Base.RefValue{Val{-1}}) = x # enables specialized reciprocal definition
__materialize(x::Base.RefValue{Val{2}}) = x # enables specialized square definition
__materialize(x::Base.RefValue{Val{V}}) where {V} = NDArray(V) # Use binary_op POWER for other literal powers
__materialize(x::Base.RefValue) = x

# Catch unknown things...
__materialize(x) = error("Unrecognized leaf in broadcast expression: $(x)")

# Use Base for scalar-only broadcasts, including `literal_pow` wrappers.
@inline function __materialize(bc::Broadcasted{<:DefaultArrayStyle{0}})
    return Base.materialize(bc)
end

function __materialize(bc::Broadcasted{<:NDArrayStyle})
    bc = Base.Broadcast.instantiate(bc)
    return unravel_broadcast_tree(bc)
end

# The C API is binary, so evaluate flattened `+` and `*` chains pairwise.
function _unravel_flattened_associative(f, args::Tuple)
    acc = first(args)
    owns_acc = false
    for arg in Base.tail(args)
        next = try
            __materialize(Base.broadcasted(f, acc, arg))
        finally
            owns_acc && acc isa NDArray && destroy!(acc)
        end
        acc = next
        owns_acc = acc isa NDArray
    end
    return acc
end

# Destroy promote copies and non-leaf materialized NDArrays (nested results / Val{V}).
@inline function _destroy_unfused_arg_temps!(orig, materialized, promoted)
    if promoted isa NDArray && promoted !== materialized
        destroy!(promoted)
    end
    if materialized isa NDArray && !(orig isa NDArray)
        destroy!(materialized)
    end
    return nothing
end

# Un-fused implementation of broadcast tree
function unravel_broadcast_tree(bc::Broadcasted)
    if length(bc.args) > 2 && _is_flattened_associative(bc.f)
        return _unravel_flattened_associative(bc.f, bc.args)
    end

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
    result = __broadcast(bc.f, out, in_args...)
    for i in eachindex(materialized_args)
        _destroy_unfused_arg_temps!(bc.args[i], materialized_args[i], in_args[i])
    end
    return result
end

# Slice destinations must assign into their parent store.
@inline function _store_broadcast_result!(
    dest::NDArray{T}, temp_result::NDArray{T}
) where {T}
    if _is_ndarray_slice(dest)
        nda_assign(dest, temp_result)
        destroy!(temp_result)
    else
        nda_move(dest, temp_result)
    end
    return dest
end

@inline _copyto_unfused!(dest::NDArray{T}, temp_result::NDArray{T}) where {T} =
    _store_broadcast_result!(
        dest, temp_result
    )

@inline function _copyto_unfused!(dest::NDArray{T}, temp_result::NDArray) where {T}
    promoted = checked_promote_arr(temp_result, T)
    _store_broadcast_result!(dest, promoted)
    destroy!(temp_result)
    return dest
end

# Number of nested `Broadcasted` nodes (ops) in the pre-flatten tree.
@inline _broadcast_tree_length(@nospecialize(_)) = 0
@inline _broadcast_tree_length(bc::Broadcasted) =
    1 + _broadcast_tree_length_args(bc.args)
@inline _broadcast_tree_length_args(::Tuple{}) = 0
@inline function _broadcast_tree_length_args(args::Tuple)
    return _broadcast_tree_length(getfield(args, 1)) +
           _broadcast_tree_length_args(Base.tail(args))
end

# A single native operation uses the C API. Unknown functions need the GPU
# broadcast kernel even when the preference normally skips single-op fusion.
@inline _has_unfused_broadcast(f, ::Val) = false
@inline _has_unfused_broadcast(::typeof(muladd), ::Val{3}) = true
@inline _has_unfused_broadcast(::typeof(abs2), ::Val{1}) = true
@inline _has_unfused_broadcast(bc::Broadcasted) =
    _has_unfused_broadcast(bc.f, Val(length(bc.args)))

# Prefer fusion when the tree has enough ops, or a single op has no native path.
# When that const is <= 1, every Broadcasted qualifies and the length check
# compiles out (`@static`).
@inline function _should_attempt_broadcast_fusion(dest::NDArray, bc::Broadcasted)
    @static if FUSE_BROADCAST_MIN_OPS <= 1
        return can_fuse_linear_broadcast(dest, bc)
    else
        operation_count = _broadcast_tree_length(bc)
        meets_fusion_minimum = operation_count >= FUSE_BROADCAST_MIN_OPS
        single_operation_needs_fusion =
            operation_count == 1 && !_has_unfused_broadcast(bc)
        worth_fusing = meets_fusion_minimum || single_operation_needs_fusion
        worth_fusing || return false
        return can_fuse_linear_broadcast(dest, bc)
    end
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

    # Require an active GPU target so `--gpus 0` stays on the unfused path.
    # Fusion requires same-shaped NDArray leaves; otherwise fall back.
    # Single native ops below `FUSE_BROADCAST_MIN_OPS` use the unfused C API.
    @static if FUSE_BROADCAST_EXPRS
        gpu_available = _has_gpu_target()
        should_fuse = gpu_available && _should_attempt_broadcast_fusion(dest, bc)
        if should_fuse
            return fuse_broadcast_tree!(dest, bc)
        else
            return _copyto_unfused!(dest, unravel_broadcast_tree(bc))
        end
    else
        return _copyto_unfused!(dest, unravel_broadcast_tree(bc))
    end
end

# Support .=
@inline Base.copyto!(dest::NDArray, bc::Broadcasted{Nothing}) = _copyto!(dest, bc)
@inline Base.copyto!(dest::NDArray, bc::Broadcasted{<:NDArrayStyle}) = _copyto!(dest, bc)

#! TODO ADD MAP FUSED IMPLEMENTATIONS
