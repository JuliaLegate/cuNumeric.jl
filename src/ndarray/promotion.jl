
smaller_type(::Type{A}, ::Type{B}) where {A,B} = ifelse(sizeof(A) < sizeof(B), A, B)
is_wider_type(::Type{A}, ::Type{B}) where {A,B} = sizeof(A) > sizeof(B)

checked_promote_arr(arr::NDArray{T}, ::Type{T}) where {T} = arr

function checked_promote_arr(arr::NDArray{T}, ::Type{S}) where {T,S}
    return checked_promote_arr(promote_type, arr, S)
end

# `op` only names the caller in the error message, so that e.g. cholesky reports
# itself rather than `promote_type`.
checked_promote_arr(op, arr::NDArray{T}, ::Type{T}) where {T} = arr

function checked_promote_arr(op, arr::NDArray{T}, ::Type{S}) where {T,S}
    is_wider_type(S, T) && assertpromotion(op, T, S)
    return as_type(arr, S)
end

unchecked_promote_arr(arr::NDArray{T}, ::Type{T}) where {T} = arr
unchecked_promote_arr(arr::NDArray{T}, ::Type{S}) where {T,S} = as_type(arr, S)
# Unfused broadcast leaves Numbers as scalars until promote; always a fresh 0-d array.
unchecked_promote_arr(x::Number, ::Type{T}) where {T} = NDArray(T(x))

# Fusion keeps Numbers as scalars in the PTX arg buffer (no 0-d NDArray).
unchecked_promote_scalar(x::Number, ::Type{T}) where {T} = T(x)
unchecked_promote_scalar(x, ::Type) = x

# kinda hacky, but lets us support weird cases like broadcasting literal_pow
unchecked_promote_arr(::Base.RefValue{typeof(^)}, ::Type{T}) where {T} = typeof(Base.:(^))
unchecked_promote_arr(::Base.RefValue{Val{V}}, ::Type{T}) where {T,V} = Val{V}

@inline _is_flattened_associative(f) = f === (+) || f === (*)

__checked_promote_op(op, ::Type{Tuple{A}}) where {A} = __checked_promote_op(op, A)
__checked_promote_op(op, ::Type{Tuple{A,B}}) where {A,B} = __checked_promote_op(op, A, B)

# Path for literal powers
@inline function __checked_promote_op(
    f::typeof(Base.literal_pow), a::Type{Tuple{_,ARR_TYPE,Val{POWER}}}
) where {_,ARR_TYPE,POWER}
    return __checked_promote_op(Base.:(^), ARR_TYPE, typeof(POWER))
end
function __checked_promote_op(
    f::typeof(Base.literal_pow), a::Type{Tuple{_,ARR_TYPE,Val{-1}}}
) where {_,ARR_TYPE}
    return __recip_type(ARR_TYPE)
end
function __checked_promote_op(
    f::typeof(Base.literal_pow), a::Type{Tuple{_,ARR_TYPE,Val{2}}}
) where {_,ARR_TYPE}
    return ARR_TYPE
end

__checked_promote_op(::typeof(Base.inv), ::Type{Tuple{A}}) where {A} = __recip_type(A)

# Inverse always goes to Float (not Julia behavior)
__recip_type(::Type{ARR_TYPE}) where {ARR_TYPE<:AbstractFloat} = ARR_TYPE
__recip_type(::Type{Int32}) = Float32
__recip_type(::Type{Int64}) = Float64
__recip_type(::Type{Bool}) = DEFAULT_FLOAT

@inline function __checked_promote_op(op, ::Type{A}) where {A}
    T = Base.promote_op(op, A)
    is_wider_type(T, A) && assertpromotion(op, A, T)
    return T
end

@inline function __checked_promote_op(op, ::Type{A}, ::Type{A}) where {A}
    T = Base.promote_op(op, A, A)
    is_wider_type(T, A) && assertpromotion(op, A, T)
    return T
end

@inline function __checked_promote_op(op, ::Type{A}, ::Type{B}) where {A,B}
    T = Base.promote_op(op, A, B)
    S = smaller_type(A, B)
    is_wider_type(T, S) && assertpromotion(op, S, T)
    return T
end

@inline _checked_promote_associative(op, ::Type{A}, ::Type{B}) where {A,B} =
    __checked_promote_op(op, A, B)

@inline function _checked_promote_associative(
    op, ::Type{A}, ::Type{B}, ::Type{C}, rest::Type...
) where {A,B,C}
    T = __checked_promote_op(op, A, B)
    return _checked_promote_associative(op, T, C, rest...)
end

# For literal powers which are often Int64, do not check for promotion to double
# The result of promote_op with a literal integer power is always the base type
# Base.promote_op(^, Float32, Int64) == Float32
# Base.promote_op(^, Int32, Int64) == Int32
function __my_promote_type(::Type{typeof(^)}, ::Type{A}, ::Type{Val{V}}) where {A,V}
    return __checked_promote_op(Base.:(^), A, typeof(V))
end

#! Not exaclty Julia behavior, but it it makes life easier...
#! Needed to handle negative powers which may or may not convert to
#! float in Julia. We will just always convert to Float
__my_promote_type(::Type{typeof(^)}, ::Type{Int32}, ::Type{Int32}) = Float32
__my_promote_type(::Type{typeof(^)}, ::Type{Int64}, ::Type{Int64}) = Float64
__my_promote_type(::Type{typeof(^)}, ::Type{Int32}, ::Type{Int64}) = Float64
__my_promote_type(::Type{typeof(^)}, ::Type{Int64}, ::Type{Int32}) = Float64
__my_promote_type(::Type{typeof(^)}, ::Type{Bool}, ::Type{Int32}) = Float32
__my_promote_type(::Type{typeof(^)}, ::Type{Bool}, ::Type{Int64}) = Float64

__my_promote_type(::Type{A}) where {A} = A
__my_promote_type(::Type{A}, ::Type{A}) where {A} = A

@inline function __my_promote_type(::Type{A}, ::Type{B}) where {A,B}
    T = promote_type(A, B)
    S = smaller_type(A, B)
    is_wider_type(T, S) && assertpromotion(promote_type, S, T)
    return T
end

# Flattened fusion (and any n-ary leaf list) folds pairwise.
@inline function __my_promote_type(::Type{A}, ::Type{B}, ::Type{C}, rest::Type...) where {A,B,C}
    return __my_promote_type(__my_promote_type(A, B), C, rest...)
end
