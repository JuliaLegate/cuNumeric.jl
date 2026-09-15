# Mapping and reduction policies are separate: sum/prod widen small integers,
# whereas mapreduce with +/* uses the ordinary scalar operators.
struct NoReductionInit end
const _MR_ADD = Union{typeof(+),typeof(Base.add_sum)}
const _MR_MUL = Union{typeof(*),typeof(Base.mul_prod)}
const _MR_EXTREMA = Union{typeof(min),typeof(max)}
const _MR_OP = Union{_MR_ADD,_MR_MUL,_MR_EXTREMA}
const _MR_DIMS = Union{Integer,Tuple}

_mr_operator(op::_MR_OP) = op
_mr_operator(op) = throw(ArgumentError("mapreduce supports only +, *, min, and max"))
_mr_combine(::_MR_ADD) = (+)
_mr_combine(::_MR_MUL) = (*)
_mr_combine(op::_MR_EXTREMA) = op
_mr_redop(::_MR_ADD, ::Type) = MAPREDUCE_ADD
_mr_redop(::_MR_MUL, ::Type) = MAPREDUCE_MUL
_mr_redop(::typeof(min), ::Type) = MAPREDUCE_MIN
_mr_redop(::typeof(max), ::Type) = MAPREDUCE_MAX
_mr_redop(::_MR_MUL, ::Type{Bool}) = MAPREDUCE_AND
_mr_redop(::typeof(min), ::Type{Bool}) = MAPREDUCE_AND
_mr_redop(::typeof(max), ::Type{Bool}) = MAPREDUCE_OR

_mr_storage(::_MR_OP, ::Type{T}) where {T} = T
_mr_storage(::_MR_EXTREMA, ::Type{Float32}) = UInt32
_mr_storage(::_MR_EXTREMA, ::Type{Float64}) = UInt64
_mr_encode(::_MR_OP, x) = x
_mr_nan_key(::typeof(min), ::Type{U}) where {U} = zero(U)
_mr_nan_key(::typeof(max), ::Type{U}) where {U} = typemax(U)
@inline function _mr_encode(op::_MR_EXTREMA, x::T) where {T<:Union{Float32,Float64}}
    U = _mr_storage(op, T)
    bits = reinterpret(U, x)
    sign = one(U) << (8sizeof(U) - 1)
    return isnan(x) ? _mr_nan_key(op, U) : (bits & sign == 0 ? bits ⊻ sign : ~bits)
end
_mr_decode(::_MR_OP, ::Type{T}, x) where {T} = x
@inline function _mr_decode(op::_MR_EXTREMA, ::Type{T}, x::U) where {T<:Union{Float32,Float64},U<:Unsigned}
    x == _mr_nan_key(op, U) && return T(NaN)
    sign = one(U) << (8sizeof(U) - 1)
    return reinterpret(T, x & sign == 0 ? ~x : x ⊻ sign)
end

_mr_identity(::_MR_ADD, ::Type{T}) where {T} = zero(T)
# -0 is neutral even when all full-reduction inputs are negative zero.
_mr_identity(::_MR_ADD, ::Type{T}) where {T<:AbstractFloat} = -zero(T)
_mr_identity(::_MR_ADD, ::Type{Complex{T}}) where {T} = Complex{T}(-zero(T), -zero(T))
_mr_identity(::_MR_MUL, ::Type{T}) where {T} = one(T)
_mr_identity(::typeof(min), ::Type{T}) where {T} = typemax(T)
_mr_identity(::typeof(max), ::Type{T}) where {T} = typemin(T)

function _mr_dims(shape::Dims{N}, ::Colon) where {N}
    return ntuple(_ -> true, max(N, 1)), ()
end
function _mr_dims(shape::Dims{N}, dims) where {N}
    region = dims isa Integer ? (dims,) : dims
    region isa Tuple || throw(ArgumentError("dims must be :, an integer, or a tuple of integers"))
    Base.reduced_indices(map(Base.OneTo, shape), region) # Base's validation, including redundant axes
    mask = ntuple(d -> d in region, max(N, 1))
    return mask, ntuple(d -> mask[d] ? 1 : shape[d], N)
end

_mr_scalar_capture(::Type{T}) where {T<:SUPPORTED_ARRAY_TYPES} = true
_mr_scalar_capture(::Type{T}) where {T} = isbitstype(T) && all(_mr_scalar_capture, fieldtypes(T)) && !isprimitivetype(T)
struct MapReduceConvert{T} end
(::MapReduceConvert{T})(x) where {T} = T(x)
_mr_callable(f) = f
_mr_callable(::Type{T}) where {T<:SUPPORTED_ARRAY_TYPES} = MapReduceConvert{T}()
function _mr_mapped_type(f::F, ::Type{T}) where {F,T}
    _mr_scalar_capture(F) || throw(ArgumentError("mapreduce requires an isbits callable with scalar captures; captured arrays and pointers are unsupported"))
    M = Base.promote_op(f, T)
    isconcretetype(M) && M <: SUPPORTED_ARRAY_TYPES || throw(ArgumentError("mapreduce mapping must infer a supported scalar result type; inferred $M"))
    return M
end

function _mr_accumulator(op::_MR_OP, ::Type{M}) where {M}
    op isa _MR_EXTREMA && M <: Complex && throw(ArgumentError("min/max mapreduce does not support complex mapped values"))
    R = Base.promote_op(Base.reduce_first, typeof(op), M)
    isconcretetype(R) && R <: SUPPORTED_ARRAY_TYPES || throw(ArgumentError("unsupported mapreduce accumulator type $R"))
    return R
end
_mr_accumulator(op, M, ::NoReductionInit, dims::_MR_DIMS) = _mr_accumulator(op, M)
_mr_accumulator(op, M, init, ::Colon) = _mr_accumulator(op, M)
function _mr_accumulator(op, M, init::I, dims::_MR_DIMS) where {I}
    # Narrowing after each update is not an associative distributed reduction.
    R = _mr_accumulator(op, M)
    op isa _MR_EXTREMA && I <: Complex && throw(ArgumentError("min/max mapreduce does not support complex init"))
    Base.promote_op(_mr_combine(op), I, R) === I || throw(ArgumentError("dimensional mapreduce requires init's type to hold the accumulator without narrowing; use init::$R"))
    return I
end

_mr_finish(op, x, ::NoReductionInit) = x
_mr_finish(op, x, init) = op(init, x)
_mr_output_type(op, R, ::NoReductionInit, dims::_MR_DIMS) = R
_mr_output_type(op, R, init, ::Colon) = Base.promote_op(_mr_combine(op), typeof(init), R)
_mr_output_type(op, R, init, dims::_MR_DIMS) = typeof(init)
_mr_output_type(op, R, ::NoReductionInit, ::Colon) = R

_mr_empty(f, op, T, M, ::NoReductionInit, ::Colon) = Base.mapreduce_empty(f, op, T)
_mr_empty(f, op, T, M, init, dims) = init
_mr_empty(f, op::_MR_ADD, T, M, ::NoReductionInit, dims::_MR_DIMS) = zero(_mr_accumulator(op, M))
_mr_empty(f, op::_MR_MUL, T, M, ::NoReductionInit, dims::_MR_DIMS) = one(_mr_accumulator(op, M))
# Base rejects empty reduced axes before scalar reduction dispatch (which can
# throw MethodError on Julia 1.10). abs/abs2 maxima have a separate zero seed.
_mr_empty(f, op::_MR_EXTREMA, T, M, ::NoReductionInit, dims::_MR_DIMS) =
    throw(ArgumentError("reducing over an empty collection is not allowed"))
_mr_empty(f::Union{typeof(abs),typeof(abs2)}, op::typeof(max), T, M, ::NoReductionInit, dims::_MR_DIMS) =
    Base.mapreduce_empty(f, op, T)

"""
    mapreduce(f, op, A::NDArray; dims=:, init)

Fuse a scalar mapping with a distributed GPU reduction. Supported operators are
`+`, `*`, `min`, and `max`. Full reductions return a 0-d `NDArray`; explicit
dimensions retain singleton axes. `sum(f, A)` and `prod(f, A)` use Base's integer
widening rules, subject to `allowpromotion`.

Requires an active GPU target and a type-stable GPU-compilable callable with only
isbits scalar captures. One input array is supported; complex results support
only addition/product. Narrowing dimensional `init` types are unsupported.
Floating-point results may differ in rounding with partitioning; extrema preserve
NaNs and signed zeros, but not NaN payloads. See the mapped-reductions documentation.
"""
function Base.mapreduce(f, op, A::NDArray{T}; dims=:, init=NoReductionInit()) where {T}
    op = _mr_operator(op)
    mask, shape = _mr_dims(size(A), dims)
    _has_gpu_target() || throw(ArgumentError("mapped reductions currently require a Legate GPU target"))
    mapper = _mr_callable(f)
    M = _mr_mapped_type(mapper, T)
    init isa NoReductionInit || (isbitstype(typeof(init)) && init isa SUPPORTED_ARRAY_TYPES) || throw(ArgumentError("init must be a supported scalar number"))
    R = _mr_accumulator(op, M, init, dims)
    O = _mr_output_type(op, R, init, dims)
    isconcretetype(O) && O <: SUPPORTED_ARRAY_TYPES || throw(ArgumentError("unsupported mapreduce output type $O"))
    is_wider_type(M, T) && assertpromotion(f, T, M)
    is_wider_type(R, M) && assertpromotion(op, M, R)
    is_wider_type(O, R) && assertpromotion(op, R, O)
    nreduce = prod(d -> mask[d] ? size(A, d) : 1, 1:ndims(A))
    if nreduce == 0
        value = _mr_empty(f, op, T, M, init, dims)
        return nda_full_array(shape, value)
    end
    isempty(A) && return nda_zeros_array(shape, O)
    return _mr_launch(mapper, op, A, R, O, mask, shape, init, dims)
end

Base.mapreduce(f, op, A::NDArray, B::AbstractArray, rest::AbstractArray...; kwargs...) =
    throw(ArgumentError("mapreduce currently supports one input NDArray"))

for (name, op) in ((:sum, Base.add_sum), (:prod, Base.mul_prod), (:minimum, min), (:maximum, max))
    @eval Base.$name(f, A::NDArray; kwargs...) = mapreduce(f, $op, A; kwargs...)
end
