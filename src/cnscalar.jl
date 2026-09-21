export CNFloat, CNInt, CNUInt, CNBool, CNReal, CNComplex, CNScalar, DeviceScalar,
    cnscalar, allowautofetch, @allowautofetch

"A floating-point device scalar; wrapping its 0D storage does not synchronize."
struct CNFloat{T<:AbstractFloat,P} <: AbstractFloat
    value::NDArray{T,0,P}
end

"A signed integer device scalar backed by a 0D NDArray."
struct CNInt{T<:Signed,P} <: Signed
    value::NDArray{T,0,P}
end

"An unsigned integer device scalar backed by a 0D NDArray."
struct CNUInt{T<:Unsigned,P} <: Unsigned
    value::NDArray{T,0,P}
end

"A Boolean device scalar. Bool is concrete, so the wrapper subtypes Integer."
struct CNBool{T<:Bool,P} <: Integer
    value::NDArray{T,0,P}
end

"A complex device scalar backed by a 0D NDArray; wrapping does not synchronize."
struct CNComplex{T<:Complex,P} <: Number
    value::NDArray{T,0,P}
end

# Bounded wrapper branches allow both DeviceScalar{Float64} and
# DeviceScalar{ComplexF64}; an exact CNComplex{Float64} would violate its bound.
const CNReal{T} = Union{CNFloat{<:T},CNInt{<:T},CNUInt{<:T},CNBool{<:T}}
const CNScalar{T} = Union{CNReal{T},CNComplex{<:T}}
const DeviceScalar{T} = Union{NDArray{T,0},CNScalar{T}}
const _REAL_SCALAR_WRAPPERS = (CNFloat, CNInt, CNUInt, CNBool)
const _SCALAR_WRAPPERS = (_REAL_SCALAR_WRAPPERS..., CNComplex)
for (W, H) in ((CNFloat, AbstractFloat), (CNInt, Signed), (CNUInt, Unsigned),
               (CNBool, Bool), (CNComplex, Complex))
    @eval cnscalar(x::NDArray{T,0}) where {T<:$H} = $W(x)
    @eval _scalar_type(::Type{T}) where {T<:$H} = $W{T}
    @eval _scalar_eltype(::Type{<:$W{T}}) where {T} = T
end
cnscalar(x::CNScalar) = x
_scalar_result(x::NDArray{<:Number,0}) = cnscalar(x)
_scalar_result(x) = x

_scale_storage(x::CNScalar) = x.value
# Host shortcuts must not inspect a device coefficient's value.
_host_iszero(::CNScalar) = false
_host_isone(::CNScalar) = false

# Numeric data stays in backend storage; host control parameters are explicitly
# permission-checked for both wrapped and unwrapped device scalars.
_maybe_fetch(x) = x
function _maybe_fetch(x::DeviceScalar{<:Real})
    _assert_allowautofetch()
    return only(_scale_storage(x))
end

_coefficient_type(x) = typeof(x)
_coefficient_type(x::DeviceScalar) = eltype(_scale_storage(x))
_coefficient_as(::Type{T}, x::Number) where {T} = convert(T, x)
_coefficient_as(::Type{T}, x::DeviceScalar) where {T} = checked_promote_arr(_scale_storage(x), T)

searchsortedfirst(a::NDArray{T,1}, x::CNScalar) where {T} = searchsortedfirst(a, x.value)
searchsortedlast(a::NDArray{T,1}, x::CNScalar) where {T} = searchsortedlast(a, x.value)

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
    allowautofetch(f, allow=true)
    allowautofetch(allow::Bool=true)
    @allowautofetch expression

Permit implicit host extraction of CNScalars for comparisons, predicates, and
conversion to host numeric types. Arithmetic stays on the backend. The do-block
and macro restore the calling task's previous permission, including on errors.
Permission is task-local and separate from scalar indexing and promotion.
This is not a fallback for arbitrary functions with unsupported argument types.
"""
allowautofetch(f::F, allow::Bool=true) where {F} =
    task_local_storage(f, :cuNumericAllowAutoFetch, allow)
allowautofetch(allow::Bool=true) = (task_local_storage(:cuNumericAllowAutoFetch, allow); nothing)

macro allowautofetch(ex)
    quote
        local previous = get(task_local_storage(), :cuNumericAllowAutoFetch, nothing)
        task_local_storage(:cuNumericAllowAutoFetch, true)
        @__tryfinally($(esc(ex)),
            if isnothing(previous)
                delete!(task_local_storage(), :cuNumericAllowAutoFetch)
            else
                task_local_storage(:cuNumericAllowAutoFetch, previous)
            end)
    end
end

function _assert_allowautofetch()
    get(task_local_storage(), :cuNumericAllowAutoFetch, false) && return nothing
    throw(ArgumentError("Implicit CNScalar host extraction is disabled. Use " *
                        "allowautofetch() do ... end or @allowautofetch, or explicitly call fetch(x)."))
end

"""
    fetch(x::CNScalar)
    fetch(x::NDArray)

Retrieve a native Julia scalar, waiting for the result as needed. An NDArray
must contain exactly one element. Explicit fetching does not require
`allowautofetch` or `allowscalar` permission.
"""
Base.fetch(x::CNScalar) = only(x.value)
Base.only(x::CNScalar) = fetch(x)
# Preserve explicit scalar indexing of reduction results, including allowscalar.
# Number's default getindex would instead return the wrapper unchanged.
Base.getindex(x::CNScalar) = x.value[]
destroy!(x::CNScalar) = destroy!(x.value)
Base.copy(x::CNScalar) = cnscalar(copy(x.value))
Base.broadcastable(x::CNScalar) = x.value
# Ref(device_scalar) still represents one backend value, not a host kernel arg.
Base.broadcastable(x::Base.RefValue{<:DeviceScalar}) = _scale_storage(x[])
# Display is an intentional host extraction, just as for the backing 0D NDArray.
Base.show(io::IO, x::CNScalar) = show(io, x.value)
Base.show(io::IO, mime::MIME"text/plain", x::CNScalar) = show(io, mime, x.value)

_scalar_operand(x::CNScalar) = x.value
_scalar_operand(x::Number) = x
_scalar_operand(x::NDArray{<:Any,0}) = x
_scalar_binary(f, x, y) = cnscalar(broadcast(f, _scalar_operand(x), _scalar_operand(y)))
function _scalar_compare(f, x, y)
    _assert_allowautofetch()
    result = broadcast(f, _scalar_operand(x), _scalar_operand(y))
    value = only(result)
    destroy!(result)
    return value
end
_scalar_host(x::CNScalar) = fetch(x)
_scalar_host(x::Number) = x
function _scalar_compare(f::Union{typeof(isless),typeof(isequal)}, x, y)
    _assert_allowautofetch()
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
Base.literal_pow(::typeof(^), x::CNScalar, ::Val{P}) where {P} = x ^ P


Base.:*(x::CNScalar, a::NDArray) = broadcast(*, x, a)
Base.:*(a::NDArray, x::CNScalar) = broadcast(*, a, x)
for op in (:+, :-, :*, :/, :^)
    @eval begin
        Base.$op(x::CNScalar, a::NDArray{<:SUPPORTED_ARRAY_TYPES,0}) = _scalar_binary($op, x, a)
        Base.$op(a::NDArray{<:SUPPORTED_ARRAY_TYPES,0}, x::CNScalar) = _scalar_binary($op, a, x)
    end
end

function _scalar_unary(f, x::CNScalar)
    # Some backend unary kernels reject rank zero. A size-one view avoids host
    # extraction and works on both CPU and GPU targets.
    input = cuNumeric.reshape(x.value, 1)
    output = broadcast(f, input)
    result = cuNumeric.reshape(output, ())
    destroy!(input)
    destroy!(output)
    return cnscalar(result)
end
for op in (:-, :abs, :sqrt), W in _SCALAR_WRAPPERS
    @eval Base.$op(x::$W) = _scalar_unary($op, x)
end
for op in (:conj, :real, :imag)
    @eval Base.$op(x::CNComplex) = _scalar_unary($op, x)
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
Base.abs2(x::CNReal) = x * x
Base.abs2(x::CNComplex) = real(x * conj(x))
Base.:!(x::CNReal{Bool}) = _scalar_unary(!, x)
for op in (:iszero, :isone, :isfinite, :isinf, :isnan), W in _SCALAR_WRAPPERS
    @eval function Base.$op(x::$W)
        _assert_allowautofetch()
        return $op(fetch(x))
    end
end

_scalar_convert(::Type{T}, x::CNScalar) where {T} = cnscalar(checked_promote_arr(x.value, T))
_scalar_convert(::Type{T}, x::Number) where {T} = cnscalar(NDArray(convert(T, x)))
function _checked_scalar_convert(::Type{T}, x::CNScalar) where {T}
    # These conversions can throw InexactError based on the value. A backend
    # dtype cast would silently truncate or discard an imaginary component.
    _assert_allowautofetch()
    return cnscalar(NDArray(convert(T, fetch(x))))
end
_scalar_convert(::Type{T}, x::CNComplex) where {T<:Real} = _checked_scalar_convert(T, x)
_scalar_convert(::Type{T}, x::CNFloat) where {T<:Integer} =
    _checked_scalar_convert(T, x)
function _scalar_convert(::Type{T}, x::Union{CNInt,CNUInt,CNBool}) where {T<:Integer}
    S = _scalar_eltype(typeof(x))
    if typemin(T) <= typemin(S) && typemax(T) >= typemax(S)
        return cnscalar(checked_promote_arr(x.value, T))
    end
    return _checked_scalar_convert(T, x)
end
for W in _SCALAR_WRAPPERS
    @eval begin
        $W{T}(x::Number) where {T} = _scalar_convert(T, x)
        Base.convert(::Type{S}, x::Number) where {T,S<:$W{T}} = _scalar_convert(T, x)
        Base.convert(::Type{S}, x::S) where {S<:$W} = x
        Base.zero(::Type{S}) where {S<:$W} = cnscalar(NDArray(zero(_scalar_eltype(S))))
        Base.one(::Type{S}) where {S<:$W} = cnscalar(NDArray(one(_scalar_eltype(S))))
    end
end
Base.zero(x::CNScalar) = zero(typeof(x))
Base.one(x::CNScalar) = one(typeof(x))
Base.real(::Type{S}) where {S<:CNScalar} = _scalar_type(real(_scalar_eltype(S)))
Base.float(x::CNScalar) = _scalar_convert(float(_scalar_eltype(typeof(x))), x)
Base.float(x::CNFloat) = x
Base.float(::Type{S}) where {S<:CNScalar} = _scalar_type(float(_scalar_eltype(S)))

for T in Base.uniontypes(SUPPORTED_ARRAY_TYPES), W in _SCALAR_WRAPPERS
    @eval function Base.convert(::Type{$T}, x::$W)
        _assert_allowautofetch()
        return convert($T, fetch(x))
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
Base.mapreduce(f, op, x::CNScalar; kwargs...) = mapreduce(f, op, x.value; kwargs...)
_div_nelem(x::CNScalar, n::Integer) = x / n
function _sqrt_ndarray(x::CNScalar)
    result = sqrt(x)
    destroy!(x)
    return result
end
