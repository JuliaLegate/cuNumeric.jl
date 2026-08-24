global const floaty_unary_ops_no_args = Dict{Function,UnaryOpCode}(
    Base.acos => cuNumeric.ARCCOS,
    Base.acosh => cuNumeric.ARCCOSH,
    Base.asin => cuNumeric.ARCSIN,
    Base.asinh => cuNumeric.ARCSINH,
    Base.atan => cuNumeric.ARCTAN,
    Base.atanh => cuNumeric.ARCTANH,
    Base.cbrt => cuNumeric.CBRT,
    Base.cos => cuNumeric.COS,
    Base.cosh => cuNumeric.COSH,
    Base.deg2rad => cuNumeric.DEG2RAD,
    Base.exp => cuNumeric.EXP,
    Base.exp2 => cuNumeric.EXP2,
    Base.expm1 => cuNumeric.EXPM1,
    Base.log => cuNumeric.LOG,
    Base.log10 => cuNumeric.LOG10,
    Base.log1p => cuNumeric.LOG1P,
    Base.log2 => cuNumeric.LOG2,
    Base.rad2deg => cuNumeric.RAD2DEG,
    Base.sin => cuNumeric.SIN,
    Base.sinh => cuNumeric.SINH,
    Base.sqrt => cuNumeric.SQRT,  # HAS SPECIAL MEANING FOR MATRIX
    Base.tan => cuNumeric.TAN,
    Base.tanh => cuNumeric.TANH,
)

global const unary_op_map_no_args = Dict{Function,UnaryOpCode}(
    Base.abs => cuNumeric.ABSOLUTE,
    # Base.conj => cuNumeric.CONJ, # handled as a special case below
    Base.:(-) => cuNumeric.NEGATIVE,
    # Base.frexp => cuNumeric.FREXP, # returns a tuple
    # missing => cuNumeric.GETARG,
    # Base.imag => cuNumeric.IMAG, # handled as a special case below
    Base.:(~) => cuNumeric.INVERT, # integers only; kernel rejects Bool
    Base.isfinite => cuNumeric.ISFINITE,
    Base.isinf => cuNumeric.ISINF,
    Base.isnan => cuNumeric.ISNAN,
    # Base.modf => cuNumeric.MODF, # returns a tuple
    # missing => cuNumeric.POSITIVE,
    Base.sign => cuNumeric.SIGN,
    Base.signbit => cuNumeric.SIGNBIT, # floats only; kernel rejects Bool/int
    Base.ceil => cuNumeric.CEIL, # floats only
    Base.floor => cuNumeric.FLOOR, # floats only
    Base.trunc => cuNumeric.TRUNC, # floats only
    # 1-arg `round.(A)` only (see __broadcast below). ROUND needs extra_args.
    Base.round => cuNumeric.RINT,
)

### SPECIAL CASES ###

# `dest .= src` lowers to `identity.(src)`. Treat identity like the native
# unary operation it is so ordinary Julia broadcast assignment works for
# NDArrays, including writable slices.
@inline function __broadcast(::typeof(identity), out::NDArray, input::NDArray)
    return nda_unary_op!(out, cuNumeric.COPY, input)
end

# Needed to support !=
Base.:(!)(input::NDArray{Bool,0}) = nda_unary_op!(similar(input), cuNumeric.LOGICAL_NOT, input)
Base.:(!)(input::NDArray{Bool,1}) = nda_unary_op!(similar(input), cuNumeric.LOGICAL_NOT, input)

# Non-broadcasted version of negation
function Base.:(-)(input::NDArray{T}) where {T}
    out = cuNumeric.zeros(T, size(input))
    return nda_unary_op!(out, cuNumeric.NEGATIVE, input)
end

function Base.real(input::NDArray{T}) where {T<:Complex}
    T_OUT = Base.promote_op(real, T)
    out = cuNumeric.zeros(T_OUT, size(input))
    return nda_unary_op!(out, cuNumeric.REAL, input)
end
Base.real(input::NDArray{<:Real}) = input

function Base.imag(input::NDArray{T}) where {T<:Complex}
    T_OUT = Base.promote_op(imag, T)
    out = cuNumeric.zeros(T_OUT, size(input))
    return nda_unary_op!(out, cuNumeric.IMAG, input)
end
Base.imag(input::NDArray{T}) where {T<:Real} = cuNumeric.zeros(T, size(input))

function Base.conj(input::NDArray{T}) where {T<:Complex}
    out = cuNumeric.zeros(T, size(input))
    return nda_unary_op!(out, cuNumeric.CONJ, input)
end
Base.conj(input::NDArray{<:Real}) = input

# Broadcoast support for complex ops
@inline function __broadcast(f::typeof(Base.real), out::NDArray, input::NDArray{<:Complex})
    return nda_unary_op!(out, cuNumeric.REAL, input)
end
@inline function __broadcast(f::typeof(Base.imag), out::NDArray, input::NDArray{<:Complex})
    return nda_unary_op!(out, cuNumeric.IMAG, input)
end
@inline function __broadcast(f::typeof(Base.conj), out::NDArray, input::NDArray{<:Complex})
    return nda_unary_op!(out, cuNumeric.CONJ, input)
end

# Fallbacks for Real types
@inline function __broadcast(f::typeof(Base.real), out::NDArray, input::NDArray{<:Real})
    # real(real_array) is just the array
    return nda_unary_op!(out, cuNumeric.COPY, input)
end
@inline function __broadcast(f::typeof(Base.imag), out::NDArray, input::NDArray{<:Real})
    # imag(real_array) is all zeros
    return nda_binary_op!(out, cuNumeric.SUBTRACT, input, input)
end
@inline function __broadcast(f::typeof(Base.conj), out::NDArray, input::NDArray{<:Real})
    # conj(real_array) is just the array
    return nda_unary_op!(out, cuNumeric.COPY, input)
end

function Base.:(-)(input::NDArray{Bool})
    promoted = checked_promote_arr(input, DEFAULT_INT)  # always new (Bool → Int)
    out = -(promoted)
    destroy!(promoted)
    return out
end

# Broadcast `.-` on Bool: Julia `-true === -1`, so promote then NEGATIVE.
@inline function __broadcast(
    ::typeof(Base.:(-)), out::NDArray{O}, input::NDArray{Bool}
) where {O<:Integer}
    assertpromotion(".-", Bool, O)
    promoted = unchecked_promote_arr(input, O)
    result = nda_unary_op!(out, cuNumeric.NEGATIVE, promoted)
    destroy!(promoted)
    return result
end

function Base.sqrt(input::NDArray{T,2}) where {T}
    return error("cuNumeric.jl does not support matrix square root.")
end

@inline function __broadcast(
    f::typeof(Base.literal_pow), out::NDArray{O}, _, input::NDArray{T}, ::Type{Val{2}}
) where {T,O}
    return nda_unary_op!(out, cuNumeric.SQUARE, input)
end

@inline function __broadcast(
    ::typeof(Base.literal_pow), out::NDArray{O}, _, input::NDArray{O}, ::Type{Val{-1}}
) where {O}
    nda_move(out, O(1) ./ input) #! REPLACE WITH RECIP ONCE FIXED
    return out
end

@inline function __broadcast(
    ::typeof(Base.literal_pow), out::NDArray{O}, _, input::NDArray, ::Type{Val{-1}}
) where {O}
    promoted = checked_promote_arr(input, O)  # always a new array when eltype ≠ O
    nda_move(out, O(1) ./ promoted) #! REPLACE WITH RECIP ONCE FIXED
    destroy!(promoted)
    return out
end

@inline function __broadcast(::typeof(Base.inv), out::NDArray{O}, input::NDArray{O}) where {O}
    nda_move(out, O(1) ./ input) #! REPLACE WITH RECIP ONCE FIXED
    return out
end

@inline function __broadcast(::typeof(Base.inv), out::NDArray{O}, input::NDArray) where {O}
    promoted = checked_promote_arr(input, O)  # always a new array when eltype ≠ O
    nda_move(out, O(1) ./ promoted) #! REPLACE WITH RECIP ONCE FIXED
    destroy!(promoted)
    return out
end

#! NEEDS TO SUPPORT inv and ^ -1
# @inline function literal_pow(::typeof(^), A::NDArray{T, 2}, ::Val{-1}) where T
#     println("HERE")
#     #! CAN WE ADD OPTIMIZATION FOR DIAGONAL MATRIX???
#     LinearAlgebra.checksquare(A)
#     out = cuNumeric.zeros(T, size(A))
#     error("Matrix inverse not supported yet")
#     # return nda_matrix_power(out, A, -1)
# end

# Only supported for Bools
@inline function __broadcast(f::typeof(Base.:(!)), out::NDArray{Bool}, input::NDArray{Bool})
    return nda_unary_op!(out, cuNumeric.LOGICAL_NOT, input)
end

# Generate hidden broadcasted version of unary ops.
for (julia_fn, op_code) in unary_op_map_no_args
    @eval begin
        @inline function __broadcast(
            f::typeof($julia_fn), out::NDArray{A}, input::NDArray{B}
        ) where {A,B}
            return nda_unary_op!(out, $(op_code), input)
        end
    end
end

# INVERT rejects Bool; on Bool, Julia `~` is the same as `!`.
@inline function __broadcast(::typeof(Base.:(~)), out::NDArray{Bool}, input::NDArray{Bool})
    return nda_unary_op!(out, cuNumeric.LOGICAL_NOT, input)
end

@noinline function _unsupported_round_broadcast()
    return throw(
        ArgumentError(
            "cuNumeric.jl only supports round.(A) (default RoundNearest / IEEE rint). " *
            "digits, sigdigits, and RoundingMode are not supported.",
        ),
    )
end

# Reject keyword forms before fusion captures Julia's keyword wrapper as a GPU callable.
@inline function Base.Broadcast.broadcasted_kwsyntax(
    ::typeof(Base.round), ::NDArray; kwargs...
)
    return _unsupported_round_broadcast()
end

# Only the 1-arg `round.(A)` method above is supported (IEEE rint / RoundNearest).
@inline function __broadcast(::typeof(Base.round), ::NDArray, ::NDArray, extra...)
    return _unsupported_round_broadcast()
end

# Some functions always return floats even when given integers
# in the case where the output is determined to be float, but
# the input is integer, we first promote the input to float.
for (julia_fn, op_code) in floaty_unary_ops_no_args
    @eval begin
        @inline function __broadcast(
            f::typeof($julia_fn), out::NDArray{T}, input::NDArray{T}
        ) where {T}
            return nda_unary_op!(out, $(op_code), input)
        end

        # If input is not already float, promote to that (temp always new → always destroy)
        @inline function __broadcast(
            f::typeof($julia_fn), out::NDArray{A}, input::NDArray{B}
        ) where {A<:SUPPORTED_FLOAT_TYPES,B<:Union{SUPPORTED_INT_TYPES,Bool}}
            promoted = checked_promote_arr(input, A)
            result = __broadcast(f, out, promoted)
            destroy!(promoted)
            return result
        end
    end
end

# CLIP / clamp needs extra_args (lo, hi). nda_unary_op! currently ccall's
# without extra scalars, so clamp is not wired. Do not revive the old
# StdVector{LegateScalar} path unless that C API exists.

@doc"""
Supported Unary Reduction Operations
===========================

The following unary reduction operations are supported and can be applied directly to `NDArray` values:

  • `all`
  • `any`
  • `maximum`
  • `minimum`
  • `prod`
  • `sum`
  • `mean`
  • `var` / `std` (sample / `corrected=true`; real types only)
  • `argmax` / `argmin` (1-d only)

Full reductions return a **0-d `NDArray`**, not a Julia scalar. Use `unwrap` or
`A[]` (with `allowscalar`) when you need a host value.

Reduction over specific dimensions is supported via the `dims` keyword argument,
following the same keepdims semantics as Julia's base reduction functions.
Multi-axis `dims=(1,2)` is implemented as sequential single-axis reductions
(the C++ kernel accepts only one axis at a time). `argmax`/`argmin` are 1-d
only (Base's N-d / `dims=` path returns `CartesianIndex`).

Examples
--------

```julia
A = cuNumeric.ones(5)

maximum(A)
sum(A)
mean(A)

# Reduce over a specific dimension
B = cuNumeric.ones(3, 4)
sum(B, dims=1)    # 1×4 result
sum(B, dims=2)    # 3×1 result

# Reduce over multiple dimensions
sum(B, dims=(1,2))  # 1×1 result
```
"""
global const unary_reduction_map = Dict{Function,UnaryRedCode}(
    # ARGMAX/ARGMIN: 1-d Base.argmax/argmin below, not this map.
    #missing => cuNumeric.CONTAINS, # strings or also integral types
    Base.maximum => cuNumeric.MAX,
    Base.minimum => cuNumeric.MIN,
    #missing => cuNumeric.NANARGMAX,
    #missing => cuNumeric.NANARGMIN,
    #missing => cuNumeric.NANMAX,
    #missing => cuNumeric.NANMIN,
    #missing => cuNumeric.NANPROD,
    Base.prod => cuNumeric.PROD,
    Base.sum => cuNumeric.SUM,
    # VARIANCE opcode is unused: compose sample var from mean / sum instead.
)

# Full reductions return 0-d NDArrays (not Julia scalars). That is intentional.

function _unary_reduction_apply(out, op_code, input::NDArray{T}, ::Type{T}) where {T}
    return nda_unary_reduction(out, op_code, input)
end

function _unary_reduction_apply(out, op_code, input::NDArray, ::Type{U}) where {U}
    promoted = unchecked_promote_arr(input, U)  # always a new array when U ≠ eltype
    result = nda_unary_reduction(out, op_code, promoted)
    destroy!(promoted)
    return result
end

function _unary_reduction_axes_apply(op_code, input::NDArray{T}, ::Type{T}, axes) where {T}
    return nda_unary_reduction_axes(op_code, input, axes, true)
end

function _unary_reduction_axes_apply(op_code, input::NDArray, ::Type{U}, axes) where {U}
    promoted = unchecked_promote_arr(input, U)  # always a new array when U ≠ eltype
    result = nda_unary_reduction_axes(op_code, promoted, axes, true)
    destroy!(promoted)
    return result
end

function _unary_reduction_impl(base_func, op_code, input::NDArray{T}, ::Colon) where {T}
    T_OUT = Base.promote_op(base_func, Vector{T})
    is_wider_type(T_OUT, T) && assertpromotion(base_func, T, T_OUT)
    out = cuNumeric.zeros(T_OUT)
    return _unary_reduction_apply(out, op_code, input, T_OUT)
end

function _unary_reduction_impl(base_func, op_code, input::NDArray{T,N}, dims::Integer) where {T,N}
    T_OUT = Base.promote_op(base_func, Vector{T})
    is_wider_type(T_OUT, T) && assertpromotion(base_func, T, T_OUT)
    axes = Int32[dims - 1]
    return _unary_reduction_axes_apply(op_code, input, T_OUT, axes)
end

# cupynumeric throws if axes.size() > 1. Compose keepdims single-axis reductions
# so `sum(A; dims=(1,2))` matches Julia's 1×1 (etc.) shape.
function _unary_reduction_impl(base_func, op_code, input::NDArray{T,N}, dims::Tuple) where {T,N}
    n = length(dims)
    n == 0 && return copy(input)
    n == 1 && return _unary_reduction_impl(base_func, op_code, input, dims[1])
    result = input
    owned = false
    for d in dims
        next = _unary_reduction_impl(base_func, op_code, result, d)
        owned && destroy!(result)
        result = next
        owned = true
    end
    return result
end

# Generate code for all unary reductions.
for (base_func, op_code) in unary_reduction_map
    @eval begin
        function $(Symbol(base_func))(input::NDArray{T,N}; dims=Colon()) where {T,N}
            return _unary_reduction_impl($base_func, $(op_code), input, dims)
        end
    end
end

function _bool_reduction_impl(op_code, input::NDArray{Bool}, ::Colon)
    out = cuNumeric.zeros(Bool)
    return nda_unary_reduction(out, op_code, input)
end

function _bool_reduction_impl(op_code, input::NDArray{Bool}, dim::Integer)
    return nda_unary_reduction_axes(op_code, input, Int32[dim - 1], true)
end

function _bool_reduction_impl(op_code, input::NDArray{Bool}, dims::Tuple)
    n = length(dims)
    n == 0 && return copy(input)
    n == 1 && return _bool_reduction_impl(op_code, input, dims[1])
    result = input
    owned = false
    for d in dims
        next = _bool_reduction_impl(op_code, result, d)
        owned && destroy!(result)
        result = next
        owned = true
    end
    return result
end

function _bool_reduction_impl(op_code, input::NDArray{Bool}, dims)
    return _bool_reduction_impl(op_code, input, Tuple(dims))
end

function Base.all(input::NDArray{Bool}; dims=Colon())
    return _bool_reduction_impl(cuNumeric.ALL, input, dims)
end

function Base.any(input::NDArray{Bool}; dims=Colon())
    return _bool_reduction_impl(cuNumeric.ANY, input, dims)
end

# Compare on-device against `zero(T)` / `_eye(T, n)` (identity filled with `one(T)`).
# Returns a 0D `NDArray{Bool}` — not a Julia `Bool`.
function Base.iszero(A::NDArray{T}) where {T}
    return all(A .== zero(T))
end
function Base.isone(A::NDArray{T,2}) where {T}
    m, n = size(A)
    m != n && return NDArray(false) # LinearAlgebra.isone: only square matrices
    return all(A .== _eye(T, m))
end

# Boolean multiplication is logical conjunction. cuPyNumeric's PROD reduction
# uses a numeric fill identity, which Legate rejects for a Boolean target.
function Base.prod(input::NDArray{Bool}; dims=Colon())
    return _unary_reduction_impl(Base.prod, cuNumeric.ALL, input, dims)
end

# Number of elements a reduction with `dims` collapses. Used by mean/var/std.
_reduction_nelem(arr::NDArray, ::Colon) = Int(prod(size(arr)))
_reduction_nelem(arr::NDArray, dim::Integer) = Int(size(arr, dim))
function _reduction_nelem(arr::NDArray, dims::Tuple)
    n = 1
    for d in dims
        n *= Int(size(arr, d))
    end
    return n
end

# Divide an NDArray by a count without going through 0-d broadcast, which
# unwraps to a Julia scalar in Broadcast.copy.
function _div_nelem(arr::NDArray{T}, n::Integer) where {T}
    FT = float(T)
    return (FT(1) / FT(n)) * arr
end

"""
    mean(A::NDArray; dims=:)

Arithmetic mean of `A`. Full reduction returns a 0-d `NDArray`, not a Julia
scalar. With `dims`, the reduced axes are kept as size 1, matching Base.
"""
function mean(arr::NDArray; dims=Colon())
    s = sum(arr; dims=dims)
    result = _div_nelem(s, _reduction_nelem(arr, dims))
    destroy!(s)
    return result
end

"""
    var(A::NDArray; corrected=true, mean=nothing, dims=:)
    std(A::NDArray; corrected=true, mean=nothing, dims=:)

Sample variance and standard deviation (`corrected=true`, divisor `n-1`),
matching Julia / StatsBase. Real types only. Returns a 0-d or reduced
`NDArray`, not a Julia scalar.
"""
function var(arr::NDArray{T}; corrected::Bool=true, mean=nothing, dims=Colon()) where {T<:Real}
    μ = isnothing(mean) ? cuNumeric.mean(arr; dims=dims) : mean
    centered = arr .- μ
    isnothing(mean) && μ isa NDArray && destroy!(μ)
    sq = centered .^ 2
    destroy!(centered)
    s = sum(sq; dims=dims)
    destroy!(sq)
    n = _reduction_nelem(arr, dims)
    denom = corrected ? n - 1 : n
    result = _div_nelem(s, denom)
    destroy!(s)
    return result
end

function std(arr::NDArray{T}; corrected::Bool=true, mean=nothing, dims=Colon()) where {T<:Real}
    return _sqrt_ndarray(var(arr; corrected=corrected, mean=mean, dims=dims))
end

# SQRT kernel rejects 0-d (shape [] vs [1]). Wrap the host sqrt back into a 0-d array.
function _sqrt_ndarray(v::NDArray{T,0}) where {T}
    s = T(sqrt(unwrap(v)))
    destroy!(v)
    return NDArray(s)
end

function _sqrt_ndarray(v::NDArray{T}) where {T}
    out = similar(v)
    nda_unary_op!(out, cuNumeric.SQRT, v)
    destroy!(v)
    return out
end

# function _count_nonzero(input::NDArray, dims)
#     nz = input .!= zero(eltype(input))
#     result = sum(nz; dims=dims)
#     destroy!(nz)
#     return result
# end
#
# """
#     count(A::NDArray{Bool}; dims=:)
#     count(!iszero, A::NDArray; dims=:)
#
# Count `true` values in a `Bool` array, or nonzeros in a numeric array.
# Returns a 0-d or reduced `NDArray` of integers, not a Julia `Int`.
# """
# function count(arr::NDArray{Bool}; dims=Colon())
#     return _count_nonzero(arr, dims)
# end
#
# function count(::ComposedFunction{typeof(!),typeof(iszero)}, arr::NDArray; dims=Colon())
#     return _count_nonzero(arr, dims)
# end

# Kernel ARGMAX/ARGMIN are 0-based. Julia indices are 1-based.
function _indices_to_one_based(raw::NDArray{Int64})
    result = nda_add_scalar(raw, Int64(1))
    destroy!(raw)
    return result
end

"""
    argmax(A::NDArray{<:Any,1})

1-based index of the first extremum, as a 0-d `NDArray{Int64}` (not a Julia
`Int`). 1-d only. Complex arrays are not supported.
"""
function argmax(arr::NDArray{T,1}) where {T}
    T <: Complex && throw(ArgumentError("argmax/argmin are not supported for complex arrays"))
    raw = nda_unary_reduction_axes(cuNumeric.ARGMAX, arr, Int32[], false)
    return _indices_to_one_based(raw)
end

"""
    argmin(A::NDArray{<:Any,1})

1-based index of the first extremum, as a 0-d `NDArray{Int64}` (not a Julia
`Int`). 1-d only. Complex arrays are not supported.
"""
function argmin(arr::NDArray{T,1}) where {T}
    T <: Complex && throw(ArgumentError("argmax/argmin are not supported for complex arrays"))
    raw = nda_unary_reduction_axes(cuNumeric.ARGMIN, arr, Int32[], false)
    return _indices_to_one_based(raw)
end

# function Base.reduce(f::Function, arr::NDArray)
#     return f(arr)
# end

#* TODO Overload broadcasting to just call this
#* e.g. sin.(ndarray) should call this or the proper generated func
function Base.map(f::Function, arr::NDArray)
    return f.(arr) # Will try to call one of the functions generated above
end
