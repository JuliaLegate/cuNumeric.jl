export NDFloat, NDInt, NDUInt, NDBool, NDReal, NDComplex, NDScalar, DeviceScalar,
    ndscalar, autounwrap, @autounwrap

"A floating-point device scalar; wrapping its 0D storage does not synchronize."
struct NDFloat{T<:AbstractFloat,P} <: AbstractFloat
    value::NDArray{T,0,P}
end

"A signed integer device scalar backed by a 0D NDArray."
struct NDInt{T<:Signed,P} <: Signed
    value::NDArray{T,0,P}
end

"An unsigned integer device scalar backed by a 0D NDArray."
struct NDUInt{T<:Unsigned,P} <: Unsigned
    value::NDArray{T,0,P}
end

"A Boolean device scalar. Bool is concrete, so the wrapper subtypes Integer."
struct NDBool{T<:Bool,P} <: Integer
    value::NDArray{T,0,P}
end

"A complex device scalar backed by a 0D NDArray; wrapping does not synchronize."
struct NDComplex{T<:Complex,P} <: Number
    value::NDArray{T,0,P}
end

# Bounded wrapper branches allow both DeviceScalar{Float64} and
# DeviceScalar{ComplexF64}; an exact NDComplex{Float64} would violate its bound.
const NDReal{T} = Union{NDFloat{<:T},NDInt{<:T},NDUInt{<:T},NDBool{<:T}}
const NDScalar{T} = Union{NDReal{T},NDComplex{<:T}}
const DeviceScalar{T} = Union{NDArray{T,0},NDScalar{T}}
const _REAL_SCALAR_WRAPPERS = (NDFloat, NDInt, NDUInt, NDBool)
const _SCALAR_WRAPPERS = (_REAL_SCALAR_WRAPPERS..., NDComplex)
for (W, H) in ((NDFloat, AbstractFloat), (NDInt, Signed), (NDUInt, Unsigned),
               (NDBool, Bool), (NDComplex, Complex))
    @eval ndscalar(x::NDArray{T,0}) where {T<:$H} = $W(x)
    @eval _scalar_type(::Type{T}) where {T<:$H} = $W{T}
    @eval _scalar_eltype(::Type{<:$W{T}}) where {T} = T
end
ndscalar(x::NDScalar) = x
_scalar_result(x::NDArray{<:Number,0}) = ndscalar(x)
_scalar_result(x) = x

_scale_storage(x::NDScalar) = x.value
# Host shortcuts must not inspect a device coefficient's value.
_host_iszero(::NDScalar) = false
_host_isone(::NDScalar) = false

# Numeric data stays in backend storage; host control parameters are explicitly
# permission-checked for both wrapped and unwrapped device scalars.
_host_parameter(x) = x
function _host_parameter(x::DeviceScalar{<:Real})
    _assert_autounwrap()
    return only(_scale_storage(x))
end

_coefficient_type(x) = typeof(x)
_coefficient_type(x::DeviceScalar) = eltype(_scale_storage(x))
_coefficient_as(::Type{T}, x::Number) where {T} = convert(T, x)
_coefficient_as(::Type{T}, x::DeviceScalar) where {T} = checked_promote_arr(_scale_storage(x), T)

searchsortedfirst(a::NDArray{T,1}, x::NDScalar) where {T} = searchsortedfirst(a, x.value)
searchsortedlast(a::NDArray{T,1}, x::NDScalar) where {T} = searchsortedlast(a, x.value)

function Base.fill!(a::NDArray, x::DeviceScalar)
    a .= _scale_storage(x)
    return a
end
function fill(x::DeviceScalar, dims::Dims)
    a = cuNumeric.zeros(_coefficient_type(x), dims)
    return fill!(a, x)
end
fill(x::DeviceScalar, dims::Int...) = fill(x, dims)
fill(x::DeviceScalar, dim::Int) = fill(x, (dim,))

"""
    autounwrap(f, allow=true)
    autounwrap(allow::Bool=true)
    @autounwrap expression

Permit implicit host extraction of NDScalars for comparisons, predicates, and
conversion to host numeric types. Arithmetic stays on the backend. The do-block
and macro restore the calling task's previous permission, including on errors.
Permission is task-local and separate from scalar indexing and promotion.
This is not a fallback for arbitrary functions with unsupported argument types.
"""
autounwrap(f::F, allow::Bool=true) where {F} =
    task_local_storage(f, :cuNumericAutoUnwrap, allow)
autounwrap(allow::Bool=true) = (task_local_storage(:cuNumericAutoUnwrap, allow); nothing)

macro autounwrap(ex)
    quote
        local previous = get(task_local_storage(), :cuNumericAutoUnwrap, nothing)
        task_local_storage(:cuNumericAutoUnwrap, true)
        @__tryfinally($(esc(ex)),
            if isnothing(previous)
                delete!(task_local_storage(), :cuNumericAutoUnwrap)
            else
                task_local_storage(:cuNumericAutoUnwrap, previous)
            end)
    end
end

function _assert_autounwrap()
    get(task_local_storage(), :cuNumericAutoUnwrap, false) && return nothing
    throw(ArgumentError("Implicit NDScalar host extraction is disabled. Use " *
                        "autounwrap() do ... end or @autounwrap, or explicitly call unwrap(x)."))
end

unwrap(x::NDScalar) = only(x.value)
Base.only(x::NDScalar) = unwrap(x)
destroy!(x::NDScalar) = destroy!(x.value)
Base.copy(x::NDScalar) = ndscalar(copy(x.value))
Base.broadcastable(x::NDScalar) = x.value
# Ref(device_scalar) still represents one backend value, not a host kernel arg.
Base.broadcastable(x::Base.RefValue{<:DeviceScalar}) = _scale_storage(x[])
# Display is an intentional host extraction, just as for the backing 0D NDArray.
Base.show(io::IO, x::NDScalar) = show(io, x.value)
Base.show(io::IO, mime::MIME"text/plain", x::NDScalar) = show(io, mime, x.value)

_scalar_operand(x::NDScalar) = x.value
_scalar_operand(x::Number) = x
_scalar_operand(x::NDArray{<:Any,0}) = x
_scalar_binary(f, x, y) = ndscalar(broadcast(f, _scalar_operand(x), _scalar_operand(y)))
function _scalar_compare(f, x, y)
    _assert_autounwrap()
    result = broadcast(f, _scalar_operand(x), _scalar_operand(y))
    value = only(result)
    destroy!(result)
    return value
end
_scalar_host(x::NDScalar) = unwrap(x)
_scalar_host(x::Number) = x
function _scalar_compare(f::Union{typeof(isless),typeof(isequal)}, x, y)
    _assert_autounwrap()
    return f(_scalar_host(x), _scalar_host(y))
end

# Explicit intersections avoid ambiguities with Base's Real/Complex methods.
for op in (:+, :-, :*, :/, :^), W in _SCALAR_WRAPPERS
    for H in (Real, Complex, AbstractFloat, Integer, Signed, Unsigned, Bool)
        @eval Base.$op(x::$W, y::$H) = _scalar_binary($op, x, y)
        @eval Base.$op(x::$H, y::$W) = _scalar_binary($op, x, y)
    end
    for V in _SCALAR_WRAPPERS
        @eval Base.$op(x::$W, y::$V) = _scalar_binary($op, x, y)
    end
end
for op in (:(==), :(!=), :isequal), W in _SCALAR_WRAPPERS
    for H in (Real, Complex, AbstractFloat, Integer, Signed, Unsigned, Bool)
        @eval Base.$op(x::$W, y::$H) = _scalar_compare($op, x, y)
        @eval Base.$op(x::$H, y::$W) = _scalar_compare($op, x, y)
    end
    for V in _SCALAR_WRAPPERS
        @eval Base.$op(x::$W, y::$V) = _scalar_compare($op, x, y)
    end
end
for op in (:<, :<=, :>, :>=, :isless), W in _REAL_SCALAR_WRAPPERS
    for H in (Real, AbstractFloat, Integer, Signed, Unsigned, Bool)
        @eval Base.$op(x::$W, y::$H) = _scalar_compare($op, x, y)
        @eval Base.$op(x::$H, y::$W) = _scalar_compare($op, x, y)
    end
    for V in _REAL_SCALAR_WRAPPERS
        @eval Base.$op(x::$W, y::$V) = _scalar_compare($op, x, y)
    end
end
for op in (:min, :max), W in _REAL_SCALAR_WRAPPERS
    for H in (Real, AbstractFloat, Integer, Signed, Unsigned, Bool)
        @eval Base.$op(x::$W, y::$H) = _scalar_binary($op, x, y)
        @eval Base.$op(x::$H, y::$W) = _scalar_binary($op, x, y)
    end
    for V in _REAL_SCALAR_WRAPPERS
        @eval Base.$op(x::$W, y::$V) = _scalar_binary($op, x, y)
    end
end
Base.literal_pow(::typeof(^), x::NDScalar, ::Val{P}) where {P} = x ^ P


Base.:*(x::NDScalar, a::NDArray) = broadcast(*, x, a)
Base.:*(a::NDArray, x::NDScalar) = broadcast(*, a, x)
for op in (:+, :-, :*, :/, :^)
    @eval begin
        Base.$op(x::NDScalar, a::NDArray{<:SUPPORTED_ARRAY_TYPES,0}) = _scalar_binary($op, x, a)
        Base.$op(a::NDArray{<:SUPPORTED_ARRAY_TYPES,0}, x::NDScalar) = _scalar_binary($op, a, x)
    end
end

function _scalar_unary(f, x::NDScalar)
    # Some backend unary kernels reject rank zero. A size-one view avoids host
    # extraction and works on both CPU and GPU targets.
    input = cuNumeric.reshape(x.value, 1)
    output = broadcast(f, input)
    result = cuNumeric.reshape(output, ())
    destroy!(input)
    destroy!(output)
    return ndscalar(result)
end
for op in (:-, :abs, :sqrt), W in _SCALAR_WRAPPERS
    @eval Base.$op(x::$W) = _scalar_unary($op, x)
end
for op in (:conj, :real, :imag)
    @eval Base.$op(x::NDComplex) = _scalar_unary($op, x)
end
for W in _SCALAR_WRAPPERS
    @eval Base.:+(x::$W) = x
    @eval Base.inv(x::$W) = one(eltype(x.value)) / x
end
for W in _REAL_SCALAR_WRAPPERS
    @eval Base.real(x::$W) = x
    @eval Base.conj(x::$W) = x
    @eval Base.imag(x::$W) = zero(x)
end
Base.abs2(x::NDReal) = x * x
Base.abs2(x::NDComplex) = real(x * conj(x))
Base.:!(x::NDReal{Bool}) = _scalar_unary(!, x)
for op in (:iszero, :isone, :isfinite, :isinf, :isnan), W in _SCALAR_WRAPPERS
    @eval function Base.$op(x::$W)
        _assert_autounwrap()
        return $op(unwrap(x))
    end
end

_scalar_convert(::Type{T}, x::NDScalar) where {T} = ndscalar(checked_promote_arr(x.value, T))
_scalar_convert(::Type{T}, x::Number) where {T} = ndscalar(NDArray(convert(T, x)))
function _checked_scalar_convert(::Type{T}, x::NDScalar) where {T}
    # These conversions can throw InexactError based on the value. A backend
    # dtype cast would silently truncate or discard an imaginary component.
    _assert_autounwrap()
    return ndscalar(NDArray(convert(T, unwrap(x))))
end
_scalar_convert(::Type{T}, x::NDComplex) where {T<:Real} = _checked_scalar_convert(T, x)
_scalar_convert(::Type{T}, x::NDFloat) where {T<:Integer} =
    _checked_scalar_convert(T, x)
function _scalar_convert(::Type{T}, x::Union{NDInt,NDUInt,NDBool}) where {T<:Integer}
    S = _scalar_eltype(typeof(x))
    if typemin(T) <= typemin(S) && typemax(T) >= typemax(S)
        return ndscalar(checked_promote_arr(x.value, T))
    end
    return _checked_scalar_convert(T, x)
end
for W in _SCALAR_WRAPPERS
    @eval begin
        $W{T}(x::Number) where {T} = _scalar_convert(T, x)
        Base.convert(::Type{S}, x::Number) where {T,S<:$W{T}} = _scalar_convert(T, x)
        Base.convert(::Type{S}, x::S) where {S<:$W} = x
        Base.zero(::Type{S}) where {S<:$W} = ndscalar(NDArray(zero(_scalar_eltype(S))))
        Base.one(::Type{S}) where {S<:$W} = ndscalar(NDArray(one(_scalar_eltype(S))))
    end
end
Base.zero(x::NDScalar) = zero(typeof(x))
Base.one(x::NDScalar) = one(typeof(x))
Base.real(::Type{S}) where {S<:NDScalar} = _scalar_type(real(_scalar_eltype(S)))
Base.float(x::NDScalar) = _scalar_convert(float(_scalar_eltype(typeof(x))), x)
Base.float(x::NDFloat) = x
Base.float(::Type{S}) where {S<:NDScalar} = _scalar_type(float(_scalar_eltype(S)))

for T in Base.uniontypes(SUPPORTED_ARRAY_TYPES), W in _SCALAR_WRAPPERS
    @eval function Base.convert(::Type{$T}, x::$W)
        _assert_autounwrap()
        return convert($T, unwrap(x))
    end
    @eval (::Type{$T})(x::$W) = convert($T, x)
end

# Generic promotion keeps values device-backed; host conversion is a separate,
# permission-checked operation. Real and complex wrappers remain distinct.
for W in _SCALAR_WRAPPERS
    @eval Base.promote_rule(::Type{S}, ::Type{T}) where {S<:$W,T<:Union{Real,Complex}} =
        _scalar_type(promote_type(_scalar_eltype(S), T))
    for V in _SCALAR_WRAPPERS
        @eval Base.promote_rule(::Type{S}, ::Type{T}) where {S<:$W,T<:$V} =
            _scalar_type(promote_type(_scalar_eltype(S), _scalar_eltype(T)))
    end
end
for T in Base.uniontypes(SUPPORTED_ARRAY_TYPES), W in _SCALAR_WRAPPERS
    @eval Base.promote_rule(::Type{$T}, ::Type{S}) where {S<:$W} =
        _scalar_type(promote_type($T, _scalar_eltype(S)))
end

# Internal composition of reductions consumes storage, never a host scalar.
Base.mapreduce(f, op, x::NDScalar; kwargs...) = mapreduce(f, op, x.value; kwargs...)
_div_nelem(x::NDScalar, n::Integer) = x / n
function _sqrt_ndarray(x::NDScalar)
    result = sqrt(x)
    destroy!(x)
    return result
end
