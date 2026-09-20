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
    mul!(y::NDArray, A::NDArray, x::NDArray[, α, β])

Compute `y = α * A * x + β * y` for a matrix and vector. Five-argument
matrix-matrix multiplication is also supported. Mixed integer/floating-point
inputs follow the usual promotion policy; integer-integer inputs are unsupported.
The destination must hold the promoted result and must not alias either input.
"""
function LinearAlgebra.mul!(y::NDArray{<:SUPPORTED_ARRAY_TYPES,1}, A::NDArray{<:SUPPORTED_ARRAY_TYPES,2}, x::NDArray{<:SUPPORTED_ARRAY_TYPES,1})
    return mul!(y, A, x, true, false)
end

function _linalg_mul!(C::NDArray{T}, cm, A::NDArray{TA}, am, B::NDArray{TB}, bm, α, β) where {T,TA,TB}
    required = _matmul_eltype(promote_type(TA, TB))
    promote_type(required, T) === T || throw(ArgumentError("mul! output type $T cannot hold promoted input type $required"))
    Ap = checked_promote_arr(mul!, A, T)
    Bp = checked_promote_arr(mul!, B, T)
    if iszero(α) || isempty(A) || isempty(B) || isempty(C)
        _contract_prepare(C, cm, Ap, am, Bp, bm)
        if !isempty(C)
            if iszero(β)
                fill!(C, zero(T))
            else
                C .*= convert(T, β)
            end
        end
    else
        _contract_same_type!(C, cm, Ap, am, Bp, bm, α, β)
    end
    Ap !== A && destroy!(Ap)
    Bp !== B && destroy!(Bp)
    return C
end

function LinearAlgebra.mul!(y::NDArray{<:SUPPORTED_ARRAY_TYPES,1}, A::NDArray{<:SUPPORTED_ARRAY_TYPES,2}, x::NDArray{<:SUPPORTED_ARRAY_TYPES,1}, α::Number, β::Number)
    return _linalg_mul!(y, "i", A, "ij", x, "j", α, β)
end

function LinearAlgebra.mul!(C::NDArray{<:SUPPORTED_ARRAY_TYPES,2}, A::NDArray{<:SUPPORTED_ARRAY_TYPES,2}, B::NDArray{<:SUPPORTED_ARRAY_TYPES,2}, α::Number, β::Number)
    return _linalg_mul!(C, "ij", A, "ik", B, "kj", α, β)
end

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
    dot(x::NDArray{<:Any,1}, y::NDArray{<:Any,1})

Hermitian inner product as an NDScalar, without unwrapping or synchronizing.
Conjugates the first operand for complex inputs. Numeric inputs are promoted
using the package's existing policy. Bool-Bool dot accumulates into Int.
"""
function LinearAlgebra.dot(x::NDArray{TX,1}, y::NDArray{TY,1}) where {TX<:SUPPORTED_ARRAY_TYPES,TY<:SUPPORTED_ARRAY_TYPES}
    length(x) == length(y) || throw(DimensionMismatch("dot vector lengths do not match"))
    T = _dot_eltype(promote_type(TX, TY))
    xp = checked_promote_arr(dot, x, T)
    yp = checked_promote_arr(dot, y, T)
    result = isempty(x) ? cuNumeric.zeros(T, ()) : _dot_same_type(xp, yp)
    xp !== x && destroy!(xp)
    yp !== y && destroy!(yp)
    return ndscalar(result)
end

_norm_nonzero(v) = ifelse(iszero(v), zero(real(v)), one(real(v)))

"""
    norm(x::NDArray, p::Real=2)

Entrywise p-norm as an NDReal (not the matrix operator norm).
The result stays on the backend; no reduction is unwrapped. Like cuPyNumeric,
powers are accumulated without scaling and may overflow or underflow. Integer
inputs convert to floating point under the existing promotion policy.
Uses mapped reductions, which currently require a GPU target.
"""
function LinearAlgebra.norm(x::NDArray{T}, p::Real=2) where {T<:_LA_FLOAT}
    R = real(T)
    isempty(x) && return ndscalar(cuNumeric.zeros(R, ()))
    p == 0 && return sum(_norm_nonzero, x)
    p == 1 && return sum(abs, x)
    p == Inf && return maximum(abs, x)
    p == -Inf && return minimum(abs, x)
    isnan(p) && return ndscalar(NDArray(R(NaN)))
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
    axpy!(α, x::NDArray{<:Any,1}, y::NDArray{<:Any,1})
    axpby!(α, x::NDArray{<:Any,1}, β, y::NDArray{<:Any,1})

Update numeric vectors with backend broadcasts and return `y`. Vector lengths
must match. Exact self-aliasing is supported; partially overlapping views are not.
"""
function LinearAlgebra.axpy!(α::Number, x::NDArray{<:SUPPORTED_ARRAY_TYPES,1}, y::NDArray{<:SUPPORTED_ARRAY_TYPES,1})
    length(x) == length(y) || throw(DimensionMismatch("axpy! vector lengths do not match"))
    iszero(α) && return y
    y .= α .* x .+ y
    return y
end

function LinearAlgebra.axpby!(α::Number, x::NDArray{<:SUPPORTED_ARRAY_TYPES,1}, β::Number, y::NDArray{<:SUPPORTED_ARRAY_TYPES,1})
    length(x) == length(y) || throw(DimensionMismatch("axpby! vector lengths do not match"))
    iszero(α) && isone(β) && return y
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

function LinearAlgebra.ldiv!(y::NDArray{<:SUPPORTED_ARRAY_TYPES,1}, D::DiagonalNDArray{<:SUPPORTED_ARRAY_TYPES}, x::NDArray{<:SUPPORTED_ARRAY_TYPES,1})
    length(x) == length(y) == size(D, 1) || throw(DimensionMismatch("diagonal solve dimensions do not match"))
    y .= x ./ _diag_vec(D)
    return y
end
