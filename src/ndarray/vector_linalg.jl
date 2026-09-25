# LinearAlgebra interfaces retain asynchronous NDArray reduction results.
const _LA_FLOAT = Union{SUPPORTED_FLOAT_TYPES,SUPPORTED_COMPLEX_TYPES}
const _LA_INTEGER = Union{SUPPORTED_INT_TYPES,Bool}

_matmul_eltype(::Type{T}) where {T<:_LA_FLOAT} = T
function _matmul_eltype(::Type{T}) where {T<:_LA_INTEGER}
    throw(ArgumentError(
        "NDArray matrix multiplication does not support integer-integer operands (including Bool); " *
        "the inputs promote to $T. Convert an operand to a floating-point type explicitly.",
    ))
end

"""
    mul!(y::NDArray, A::NDArray, x::NDArray)

Store the matrix-vector product `A * x` in `y`.
"""
function LinearAlgebra.mul!(y::NDArray{<:SUPPORTED_ARRAY_TYPES,1}, A::NDArray{<:SUPPORTED_ARRAY_TYPES,2}, x::NDArray{<:SUPPORTED_ARRAY_TYPES,1})
    return mul!(y, A, x, true, false)
end

_mul_output_scale(x::Number, ::Type{T}) where {T} = convert(T, x)
_mul_output_scale(x::NDArray, ::Type{T}) where {T} = x

function _linalg_mul!(C::NDArray{T}, cm, A::NDArray{TA}, am, B::NDArray{TB}, bm, α, β) where {T,TA,TB}
    required = _matmul_eltype(promote_type(TA, TB))
    promote_type(required, T) === T || throw(ArgumentError("mul! output type $T cannot hold promoted input type $required"))
    Ap = checked_promote_arr(mul!, A, T)
    Bp = checked_promote_arr(mul!, B, T)
    α = _scale_storage(α)
    β = _scale_storage(β)
    if _host_iszero(α) || isempty(A) || isempty(B) || isempty(C)
        _contract_prepare(C, cm, Ap, am, Bp, bm)
        if !isempty(C)
            if _host_iszero(β)
                fill!(C, zero(T))
            else
                C .*= _mul_output_scale(β, T)
            end
        end
    else
        _contract_same_type!(C, cm, Ap, am, Bp, bm, α, β)
    end
    Ap !== A && destroy!(Ap)
    Bp !== B && destroy!(Bp)
    return C
end

"""
    mul!(y::NDArray, A::NDArray, x::NDArray, α, β)

Store `α * A * x + β * y` in `y`. The destination must not alias an input.
`α` and `β` accept host numbers, 0D `NDArray{T,0}` values, or `CNScalar` wrappers.
"""
function LinearAlgebra.mul!(y::NDArray{<:SUPPORTED_ARRAY_TYPES,1}, A::NDArray{<:SUPPORTED_ARRAY_TYPES,2}, x::NDArray{<:SUPPORTED_ARRAY_TYPES,1}, α::Union{Number,DeviceScalar}, β::Union{Number,DeviceScalar})
    return _linalg_mul!(y, "i", A, "ij", x, "j", α, β)
end

"""
    mul!(C::NDArray, A::NDArray, B::NDArray, α, β)

Store `α * A * B + β * C` in `C`. The destination must not alias an input.
`α` and `β` accept host numbers, 0D `NDArray{T,0}` values, or `CNScalar` wrappers.
"""
function LinearAlgebra.mul!(C::NDArray{<:SUPPORTED_ARRAY_TYPES,2}, A::NDArray{<:SUPPORTED_ARRAY_TYPES,2}, B::NDArray{<:SUPPORTED_ARRAY_TYPES,2}, α::Union{Number,DeviceScalar}, β::Union{Number,DeviceScalar})
    return _linalg_mul!(C, "ij", A, "ik", B, "kj", α, β)
end

# Resolve intersections with LinearAlgebra's host Number signatures.
LinearAlgebra.mul!(y::NDArray{<:SUPPORTED_ARRAY_TYPES,1}, A::NDArray{<:SUPPORTED_ARRAY_TYPES,2}, x::NDArray{<:SUPPORTED_ARRAY_TYPES,1}, α::Number, β::Number) =
    _linalg_mul!(y, "i", A, "ij", x, "j", α, β)
LinearAlgebra.mul!(C::NDArray{<:SUPPORTED_ARRAY_TYPES,2}, A::NDArray{<:SUPPORTED_ARRAY_TYPES,2}, B::NDArray{<:SUPPORTED_ARRAY_TYPES,2}, α::Number, β::Number) =
    _linalg_mul!(C, "ij", A, "ik", B, "kj", α, β)

function Base.:*(A::NDArray{TA,2}, x::NDArray{TX,1}) where {TA<:SUPPORTED_ARRAY_TYPES,TX<:SUPPORTED_ARRAY_TYPES}
    T = _matmul_eltype(promote_type(TA, TX))
    size(A, 2) == length(x) || throw(DimensionMismatch("matrix-vector dimensions do not match"))
    y = cuNumeric.zeros(T, size(A, 1))
    return mul!(y, A, x)
end

_dot_eltype(::Type{T}) where {T} = T
_dot_eltype(::Type{Bool}) = Int
_dot_same_type(x::NDArray{T,1}, y::NDArray{T,1}) where {T<:Real} = nda_dot(x, y)
function _dot_same_type(x::NDArray{T,1}, y::NDArray{T,1}) where {T<:Complex}
    cx = conj.(x)
    result = nda_dot(cx, y)
    destroy!(cx)
    return result
end

"""
    dot(x::NDArray, y::NDArray)

Return the vector inner product as a `CNScalar`. Complex inputs conjugate `x`.
"""
function LinearAlgebra.dot(x::NDArray{TX,1}, y::NDArray{TY,1}) where {TX<:SUPPORTED_ARRAY_TYPES,TY<:SUPPORTED_ARRAY_TYPES}
    length(x) == length(y) || throw(DimensionMismatch("dot vector lengths do not match"))
    T = _dot_eltype(promote_type(TX, TY))
    xp = checked_promote_arr(dot, x, T)
    yp = checked_promote_arr(dot, y, T)
    result = isempty(x) ? cuNumeric.zeros(T, ()) : _dot_same_type(xp, yp)
    xp !== x && destroy!(xp)
    yp !== y && destroy!(yp)
    return cnscalar(result)
end

_norm_nonzero(v) = ifelse(iszero(v), zero(real(v)), one(real(v)))

LinearAlgebra.norm(x::NDArray{<:SUPPORTED_ARRAY_TYPES}, p::NDArray{<:Real,0}) =
    norm(x, _maybe_fetch(p))

"""
    norm(x::NDArray, p::Real=2)

Return the entrywise `p`-norm as a real `CNScalar`, not a matrix operator norm.
Dense-array norms currently require a GPU. Unscaled accumulation can overflow
or underflow.
"""
function LinearAlgebra.norm(x::NDArray{T}, p::Real=2) where {T<:_LA_FLOAT}
    p = _maybe_fetch(p)
    R = real(T)
    isempty(x) && return cnscalar(cuNumeric.zeros(R, ()))
    p == 0 && return sum(_norm_nonzero, x)
    p == 1 && return sum(abs, x)
    p == Inf && return maximum(abs, x)
    p == -Inf && return minimum(abs, x)
    isnan(p) && return cnscalar(NDArray(R(NaN)))
    exponent = R(p)
    total = p == 2 ? sum(abs2, x) : sum(v -> abs(v)^exponent, x)
    # Optimization opportunity: a specialized reduction could fuse the root into
    # its final combine kernel, after all partitions' contributions are combined.
    # dims=() uses the elementwise singleton path for this 0D result: one root
    # kernel, without unwrapping or launching another full reduction.
    result = p == 2 ? mapreduce(sqrt, +, total; dims=()) :
             mapreduce(v -> v^inv(exponent), +, total; dims=())
    destroy!(total)
    return result
end

function LinearAlgebra.norm(x::NDArray{T}, p::Real=2) where {T<:_LA_INTEGER}
    xp = checked_promote_arr(norm, x, float(T))
    result = norm(xp, p)
    destroy!(xp)
    return result
end

"""
    axpy!(α, x::NDArray, y::NDArray)

Update and return `y` with `y = α * x + y`.
"""
function LinearAlgebra.axpy!(α::Union{Number,DeviceScalar}, x::NDArray{<:SUPPORTED_ARRAY_TYPES,1}, y::NDArray{<:SUPPORTED_ARRAY_TYPES,1})
    length(x) == length(y) || throw(DimensionMismatch("axpy! vector lengths do not match"))
    _host_iszero(α) && return y
    y .= α .* x .+ y
    return y
end

"""
    axpby!(α, x::NDArray, β, y::NDArray)

Update and return `y` with `y = α * x + β * y`.
"""
function LinearAlgebra.axpby!(α::Union{Number,DeviceScalar}, x::NDArray{<:SUPPORTED_ARRAY_TYPES,1}, β::Union{Number,DeviceScalar}, y::NDArray{<:SUPPORTED_ARRAY_TYPES,1})
    length(x) == length(y) || throw(DimensionMismatch("axpby! vector lengths do not match"))
    _host_iszero(α) && _host_isone(β) && return y
    y .= α .* x .+ β .* y
    return y
end

function LinearAlgebra.rmul!(x::NDArray{<:SUPPORTED_ARRAY_TYPES}, α::Number)
    x .*= α
    return x
end

function LinearAlgebra.lmul!(α::Number, x::NDArray{<:SUPPORTED_ARRAY_TYPES})
    x .= α .* x
    return x
end

# Keep the Number signatures above to resolve Base's AbstractArray/Number
# intersections. Raw device scalars share their implementation via the wrapper.
LinearAlgebra.rmul!(x::NDArray{<:SUPPORTED_ARRAY_TYPES}, α::NDArray{<:Any,0}) = rmul!(x, cnscalar(α))
LinearAlgebra.lmul!(α::NDArray{<:Any,0}, x::NDArray{<:SUPPORTED_ARRAY_TYPES}) = lmul!(cnscalar(α), x)

function LinearAlgebra.ldiv!(y::NDArray{<:SUPPORTED_ARRAY_TYPES,1}, D::DiagonalNDArray{<:SUPPORTED_ARRAY_TYPES}, x::NDArray{<:SUPPORTED_ARRAY_TYPES,1})
    length(x) == length(y) == size(D, 1) || throw(DimensionMismatch("diagonal solve dimensions do not match"))
    y .= x ./ _diag_vec(D)
    return y
end
