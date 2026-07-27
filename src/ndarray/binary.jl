# Still missing:
#     # Base.copysign => cuNumeric.COPYSIGN, #* ANNOYING TO TEST
#     #missing => cuNumeric.fmod, #same as mod in Julia?
#     # Base.isapprox => cuNumeric.ISCLOSE, #* HANDLE rtol, atol kwargs!!!
#     # Base.ldexp => cuNumeric.LDEXP, #* LHS FLOATS, RHS INTS
#     #missing => cuNumeric.LOGADDEXP,
#     #missing => cuNumeric.LOGADDEXP2,
#     #missing => cuNumeric.NEXTAFTER,

# Binary ops which are equivalent to Julia's broadcast syntax
global const binary_op_map = Dict{Function,BinaryOpCode}(
    Base.:+ => cuNumeric.ADD,
    Base.:* => cuNumeric.MULTIPLY,
    Base.:(-) => cuNumeric.SUBTRACT,
    Base.:(^) => cuNumeric.POWER, #! SOME WEIRD EDGE CASES
    # Base.:^ => cuNumeric.FLOAT_POWER, # DONT THINK THIS IS WHAT WE WANT
    Base.max => cuNumeric.MAXIMUM,
    Base.min => cuNumeric.MINIMUM,
    Base.:(<) => cuNumeric.LESS, #* Julia also has non-broadcasted versions required `isless`
    Base.:(<=) => cuNumeric.LESS_EQUAL,  #* Julia also has non-broadcasted versions required `isless`
    Base.:(>) => cuNumeric.GREATER, #* Julia also has non-broadcasted versions required `isless`
    Base.:(>=) => cuNumeric.GREATER_EQUAL, #* Julia also has non-broadcasted versions required `isless`
    Base.:(!=) => cuNumeric.NOT_EQUAL, #*  BE SURE TO DEFINE NON-BROADCASTED VERSION (BINARY_REDUCTION)
    Base.:(==) => cuNumeric.EQUAL, #*  BE SURE TO DEFINE NON-BROADCASTED VERSION (BINARY_REDUCTION),
    Base.lcm => cuNumeric.LCM,
    Base.gcd => cuNumeric.GCD,
    # Base.xor => cuNumeric.LOGICAL_XOR, #! DO LATER
    # Base.:⊻ => cuNumeric.LOGICAL_XOR, #! DO LATER
    # Base.div => cuNumeric.FLOOR_DIVIDE, #! THESE ARE IN-EXACT FOR INTS?
    # Base.:(÷) => cuNumeric.FLOOR_DIVIDE, #! THESE ARE IN-EXACT FOR INTS?
    # Base.:(>>) => cuNumeric.RIGHT_SHIFT, #! DO LATER
    # Base.:(<<) => cuNumeric.LEFT_SHIFT, #! DO LATER
    # Base.:(&&) => (cuNumeric.LOGICAL_AND, Bool, :same_as_input), #! CANNOT OVERLOAD WTF? (see Base.andand)
    # Base.:(||) => (cuNumeric.LOGICAL_OR, Bool, :same_as_input), #! CANNOT OVERLOAD WTF?
)

global const floaty_binary_op_map = Dict{Function,BinaryOpCode}(
    Base.:/ => cuNumeric.DIVIDE,
    Base.hypot => cuNumeric.HYPOT,
    Base.atan => cuNumeric.ARCTAN2,
)

## SPECIAL CASES ##
# Promote into out's eltype, then destroy any new temps (dispatch; no runtime !==).
@inline function _nda_binary_op_promoted!(
    out::NDArray{T}, op, rhs1::NDArray{T}, rhs2::NDArray{T}
) where {T}
    return nda_binary_op!(out, op, rhs1, rhs2)
end
function _nda_binary_op_promoted!(out::NDArray{T}, op, rhs1::NDArray, rhs2::NDArray{T}) where {T}
    p1 = unchecked_promote_arr(rhs1, T)  # always new when eltype ≠ T
    result = nda_binary_op!(out, op, p1, rhs2)
    destroy!(p1)
    return result
end
function _nda_binary_op_promoted!(out::NDArray{T}, op, rhs1::NDArray{T}, rhs2::NDArray) where {T}
    p2 = unchecked_promote_arr(rhs2, T)
    result = nda_binary_op!(out, op, rhs1, p2)
    destroy!(p2)
    return result
end
function _nda_binary_op_promoted!(out::NDArray{T}, op, rhs1::NDArray, rhs2::NDArray) where {T}
    p1 = unchecked_promote_arr(rhs1, T)
    p2 = unchecked_promote_arr(rhs2, T)
    result = nda_binary_op!(out, op, p1, p2)
    destroy!(p1)
    destroy!(p2)
    return result
end

@inline function _nda_three_dot_promoted!(
    rhs1::NDArray{T}, rhs2::NDArray{T}, out::NDArray{T}
) where {T}
    return nda_three_dot_arg(rhs1, rhs2, out)
end
function _nda_three_dot_promoted!(rhs1::NDArray, rhs2::NDArray{T}, out::NDArray{T}) where {T}
    p1 = unchecked_promote_arr(rhs1, T)
    result = nda_three_dot_arg(p1, rhs2, out)
    destroy!(p1)
    return result
end
function _nda_three_dot_promoted!(rhs1::NDArray{T}, rhs2::NDArray, out::NDArray{T}) where {T}
    p2 = unchecked_promote_arr(rhs2, T)
    result = nda_three_dot_arg(rhs1, p2, out)
    destroy!(p2)
    return result
end
function _nda_three_dot_promoted!(rhs1::NDArray, rhs2::NDArray, out::NDArray{T}) where {T}
    p1 = unchecked_promote_arr(rhs1, T)
    p2 = unchecked_promote_arr(rhs2, T)
    result = nda_three_dot_arg(p1, p2, out)
    destroy!(p1)
    destroy!(p2)
    return result
end

@inline function _nda_three_dot_checked!(
    rhs1::NDArray{T}, rhs2::NDArray{T}, out::NDArray{T}
) where {T}
    return nda_three_dot_arg(rhs1, rhs2, out)
end
function _nda_three_dot_checked!(rhs1::NDArray, rhs2::NDArray{T}, out::NDArray{T}) where {T}
    p1 = checked_promote_arr(rhs1, T)
    result = nda_three_dot_arg(p1, rhs2, out)
    destroy!(p1)
    return result
end
function _nda_three_dot_checked!(rhs1::NDArray{T}, rhs2::NDArray, out::NDArray{T}) where {T}
    p2 = checked_promote_arr(rhs2, T)
    result = nda_three_dot_arg(rhs1, p2, out)
    destroy!(p2)
    return result
end
function _nda_three_dot_checked!(rhs1::NDArray, rhs2::NDArray, out::NDArray{T}) where {T}
    p1 = checked_promote_arr(rhs1, T)
    p2 = checked_promote_arr(rhs2, T)
    result = nda_three_dot_arg(p1, p2, out)
    destroy!(p1)
    destroy!(p2)
    return result
end

# Do not need broadcast operation when same shape
function Base.:(-)(rhs1::NDArray{A,N}, rhs2::NDArray{B,N}) where {A,B,N}
    promote_shape(size(rhs1), size(rhs2))
    T_OUT = __checked_promote_op(-, A, B)
    out = cuNumeric.zeros(T_OUT, size(rhs1))
    return _nda_binary_op_promoted!(out, cuNumeric.SUBTRACT, rhs1, rhs2)
end

# Do not need broadcast operation when same shape
function Base.:(+)(rhs1::NDArray{A,N}, rhs2::NDArray{B,N}) where {A,B,N}
    promote_shape(size(rhs1), size(rhs2))
    T_OUT = __checked_promote_op(+, A, B)
    out = cuNumeric.zeros(T_OUT, size(rhs1))
    return _nda_binary_op_promoted!(out, cuNumeric.ADD, rhs1, rhs2)
end

Base.:(*)(val::V, arr::NDArray{A}) where {A,V} = _mul_scalar(__my_promote_type(A, V), val, arr)
Base.:(*)(arr::NDArray{A}, val::V) where {A,V} = val * arr

_mul_scalar(::Type{T}, val, arr::NDArray{T}) where {T} = nda_multiply_scalar(arr, T(val))
function _mul_scalar(::Type{U}, val, arr::NDArray) where {U}
    promoted = unchecked_promote_arr(arr, U)  # always a new array when U ≠ eltype
    out = nda_multiply_scalar(promoted, U(val))
    destroy!(promoted)
    return out
end

function Base.:(*)(rhs1::NDArray{A,2}, rhs2::NDArray{B,2}) where {A,B}
    size(rhs1, 2) == size(rhs2, 1) ||
        throw(DimensionMismatch("Matrix dimensions incompatible: $(size(rhs1)) × $(size(rhs2))"))
    T = __my_promote_type(A, B)
    out = cuNumeric.zeros(T, (size(rhs1, 1), size(rhs2, 2)))
    return _nda_three_dot_promoted!(rhs1, rhs2, out)
end

function Base.:(*)(rhs1::NDArray{Bool,2}, rhs2::NDArray{Bool,2})
    throw(
        ArgumentError("cuNumeric.jl does not support matrix multiplication of two Boolean arrays")
    )
end

function Base.:(*)(rhs1::NDArray{<:Integer,2}, rhs2::NDArray{<:Integer,2})
    #* this is a stupid.....
    throw(
        ArgumentError("cuNumeric.jl does not support matrix multiplication of two Integer arrays")
    )
end

@doc"""
    LinearAlgebra.mul!(out::NDArray, arr1::NDArray, arr2::NDArray)

Compute the matrix multiplication of `arr1` and `arr2`, storing the result in `out`.

This function performs the operation in-place, modifying `out`.

# Examples
```@repl
a = cuNumeric.ones(2, 3)
b = cuNumeric.ones(3, 2)
out = cuNumeric.zeros(2, 2)
LinearAlgebra.mul!(out, a, b)
```
"""
function LinearAlgebra.mul!(
    out::NDArray{T,2}, rhs1::NDArray{A,2}, rhs2::NDArray{B,2}
) where {T<:SUPPORTED_NUMERIC_TYPES,A,B}
    size(rhs1, 2) == size(rhs2, 1) ||
        throw(DimensionMismatch("Matrix dimensions incompatible: $(size(rhs1)) × $(size(rhs2))"))
    (size(out, 1) == size(rhs1, 1) && size(out, 2) == size(rhs2, 2)) || throw(
        DimensionMismatch(
            "mul! output is $(size(out)), but inputs would produce $(size(rhs1,1))×$(size(rhs2,2))"
        ),
    )

    T_REQUIRED = __my_promote_type(A, B)
    if promote_type(T_REQUIRED, T) != T
        if (T_REQUIRED <: Complex && !(T <: Complex))
            throw(
                ArgumentError(
                    "Implicit promotion: mul! output has real type $(T), but inputs promote to complex type: $(T_REQUIRED)"
                ),
            )
        elseif (T_REQUIRED <: AbstractFloat && T <: Integer)
            throw(
                ArgumentError(
                    "Implicit promotion: mul! output has integer type $(T), but inputs promote to floating point type: $(T_REQUIRED)"
                ),
            )
        end
        # General case (e.g. Float64 result into Float32)
        throw(
            ArgumentError(
                "mul! output type $(T) cannot hold the promoted input type $(T_REQUIRED). Implicit promotion to wider type or complex result is disallowed."
            ),
        )
    end
    return _nda_three_dot_checked!(rhs1, rhs2, out)
end

function LinearAlgebra.mul!(out::NDArray, rhs1::NDArray{Bool,2}, rhs2::NDArray{Bool,2})
    #* Could just promote both inputs to Int32
    throw(
        ArgumentError("cuNumeric.jl does not support matrix multiplication of two Boolean arrays")
    )
end

function LinearAlgebra.mul!(out::NDArray, rhs1::NDArray{<:Integer,2}, rhs2::NDArray{<:Integer,2})
    #* this is a stupid.....
    throw(
        ArgumentError("cuNumeric.jl does not support matrix multiplication of two Integer arrays")
    )
end

# Generate hidden broadcast functions for binary ops
for (julia_fn, op_code) in binary_op_map
    @eval begin
        @inline function __broadcast(
            f::typeof($(julia_fn)), out::NDArray, rhs1::NDArray{T}, rhs2::NDArray{T}
        ) where {T}
            return nda_binary_op!(out, $(op_code), rhs1, rhs2)
        end
    end
end

# Some functions always return floats even when given integers
# in the case where the output is determined to be float, but
# the input is integer, we first promote the input to float.
for (julia_fn, op_code) in floaty_binary_op_map
    @eval begin
        @inline function __broadcast(
            f::typeof($(julia_fn)), out::NDArray, rhs1::NDArray{T}, rhs2::NDArray{T}
        ) where {T}
            return nda_binary_op!(out, $(op_code), rhs1, rhs2)
        end

        # If input is not already float, promote to that (temps always new → always destroy)
        @inline function __broadcast(
            f::typeof($(julia_fn)), out::NDArray{A}, rhs1::NDArray{B}, rhs2::NDArray{B}
        ) where {A<:SUPPORTED_FLOAT_TYPES,B<:Union{SUPPORTED_INT_TYPES,Bool}}
            p1 = checked_promote_arr(rhs1, A)
            p2 = checked_promote_arr(rhs2, A)
            result = __broadcast(f, out, p1, p2)
            destroy!(p1)
            destroy!(p2)
            return result
        end
    end
end

@inline function __broadcast(
    f::typeof(Base.:(+)), out::NDArray{O}, rhs1::NDArray{Bool}, rhs2::NDArray{Bool}
) where {O<:Integer}
    assertpromotion(".+", Bool, O)
    p1 = unchecked_promote_arr(rhs1, O)  # always new (Bool → O)
    p2 = unchecked_promote_arr(rhs2, O)
    result = nda_binary_op!(out, cuNumeric.ADD, p1, p2)
    destroy!(p1)
    destroy!(p2)
    return result
end

@inline function __broadcast(
    f::typeof(Base.:(-)), out::NDArray{O}, rhs1::NDArray{Bool}, rhs2::NDArray{Bool}
) where {O<:Integer}
    assertpromotion(".-", Bool, O)
    p1 = unchecked_promote_arr(rhs1, O)
    p2 = unchecked_promote_arr(rhs2, O)
    result = nda_binary_op!(out, cuNumeric.SUBTRACT, p1, p2)
    destroy!(p1)
    destroy!(p2)
    return result
end

# function Base.:(==)(lhs::NDArray{A}, rhs::NDArray{B}) where {A,B}
#     error("Not implemented yet")
#     #! REPLACE WITH ARRAY_EQUAL ONCE THAT IS WRAPPED
#     #! or explicit call to nda_binary_reduction
# end

# function Base.:(!=)(lhs::NDArray{A}, rhs::NDArray{B}) where {A,B}
#     error("Not implemented yet")
#     #! REPLACE WITH ARRAY_EQUAL ONCE THAT IS WRAPPED
#     #! or explicit call to nda_binary_reduction
# end

# Specializations for 2 and -1 in unary.jl
@inline function __broadcast(
    f::typeof(Base.literal_pow), out::NDArray, _, input::NDArray{T}, power::NDArray{T}
) where {T}
    return nda_binary_op!(out, cuNumeric.POWER, input, power)
end

# This is more "Julian" since a user expects map to broadcast
# their operation whereas the generated functions should technically
# only broadcast when the .() syntax is used
function Base.map(f::Function, arr1::NDArray{A,N}, arr2::NDArray{B,N}) where {A,B,N}
    return f.(arr1, arr2) # Will try to call one of the functions generated above
end

# function Base.map!(f::Function, dest::NDArray, arr1::NDArray, arr2::NDArray)
#     return f
# end
