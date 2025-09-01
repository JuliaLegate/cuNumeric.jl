@doc"""
Supported Binary Operations
===========================

The following binary operations are supported and can be applied elementwise to pairs of `NDArray` values:

  • `+`
  • `-`
  • `*`
  • `/`
  • `^`
  • `div`
  • `atan` 
  • `hypot`

These operations are applied elementwise by default and follow standard Julia semantics.

Examples
--------

```julia
A = NDArray(randn(Float64, 4))
B = NDArray(randn(Float64, 4))

A + B
A / B
hypot.(A, B)
div.(A, B)
A .^ 2
```
"""

# Still missing:
#     # Base.copysign => cuNumeric.COPYSIGN, #* ANNOYING TO TEST 
#     #missing => cuNumeric.fmod, #same as mod in Julia?
#     # Base.isapprox => cuNumeric.ISCLOSE, #* HANDLE rtol, atol kwargs!!!
#     # Base.ldexp => cuNumeric.LDEXP, #* LHS FLOATS, RHS INTS
#     #missing => cuNumeric.LOGADDEXP,
#     #missing => cuNumeric.LOGADDEXP2,
#     #missing => cuNumeric.NEXTAFTER,

# Binary ops which are equivalent to Julia's broadcast syntax
global const broadcasted_binary_op_map = Dict{Function, BinaryOpCode}(
    Base.:+ => cuNumeric.ADD,
    Base.:/ => cuNumeric.DIVIDE,
    Base.:* => cuNumeric.MULTIPLY, 
    Base.:(-) => cuNumeric.SUBTRACT,
    Base.:(^) => cuNumeric.POWER,
    # Base.:^ => cuNumeric.FLOAT_POWER, # DONT THINK THIS IS WHAT WE WANT
    Base.hypot => cuNumeric.HYPOT,
    Base.max => cuNumeric.MAXIMUM,
    Base.min => cuNumeric.MINIMUM,
    Base.:(<) => cuNumeric.LESS, #* Julia also has non-broadcasted versions required `isless`
    Base.:(<=) => cuNumeric.LESS_EQUAL,  #* Julia also has non-broadcasted versions required `isless`
    Base.:(>) => cuNumeric.GREATER, #* Julia also has non-broadcasted versions required `isless`
    Base.:(>=) => cuNumeric.GREATER_EQUAL, #* Julia also has non-broadcasted versions required `isless`
    # Base.:(!=) => (cuNumeric.NOT_EQUAL, #*  BE SURE TO DEFINE NON-BROADCASTED VERSION (BINARY_REDUCTION)
    # Base.:(==) => (cuNumeric.EQUAL, #*  BE SURE TO DEFINE NON-BROADCASTED VERSION (BINARY_REDUCTION),
    Base.atan => cuNumeric.ARCTAN2,
    Base.lcm => cuNumeric.LCM,
    Base.gcd => cuNumeric.GCD,
    Base.xor => cuNumeric.LOGICAL_XOR,
    Base.:⊻ => cuNumeric.LOGICAL_XOR,
    Base.div => cuNumeric.FLOOR_DIVIDE,
    Base.:(÷) => cuNumeric.FLOOR_DIVIDE,
    Base.:(>>) => cuNumeric.RIGHT_SHIFT,
    Base.:(<<) => cuNumeric.LEFT_SHIFT,
    # Base.:(&&) => (cuNumeric.LOGICAL_AND, Bool, :same_as_input), #! CANNOT OVERLOAD WTF? (see Base.andand)
    # Base.:(||) => (cuNumeric.LOGICAL_OR, Bool, :same_as_input), #! CANNOT OVERLOAD WTF?
)


## SPECIAL CASES ##

# Do not need broadcast operation when same shape
function Base.:(-)(rhs1::NDArray{A, N}, rhs2::NDArray{B, N}) where {A, B, N}
    # technically should call promote_shape 
    T = __my_promote_type(A, B)
    out = cuNumeric.zeros(T, size(rhs1))
    return nda_binary_op(out, cuNumeric.SUBTRACT, unchecked_promote_arr(rhs1, T), unchecked_promote_arr(rhs2, T))
end

# Do not need broadcast operation when same shape
function Base.:(+)(rhs1::NDArray{A, N}, rhs2::NDArray{B,N}) where {A, B, N}
    # technically should call promote_shape 
    T = __my_promote_type(A, B)
    out = cuNumeric.zeros(T, size(rhs1))
    return nda_binary_op(out, cuNumeric.ADD, unchecked_promote_arr(rhs1, T), unchecked_promote_arr(rhs2, T))
end

function Base.:(*)(val::V, arr::NDArray{A}) where {A, V}
    T = __my_promote_type(A, V)
    out = cuNumeric.zeros(T, size(arr))
    return nda_binary_op(out, cuNumeric.MULTIPLY, NDArray(T(val)), unchecked_promote_arr(arr, T))
end

function Base.:(*)(arr::NDArray{A}, val::V) where {A, V}
    val * arr
end

function Base.:(*)(rhs1::NDArray{A, 2}, rhs2::NDArray{B, 2}) where {A, B}
    size(rhs1, 2) == size(rhs2, 1) || throw(DimensionMismatch("Matrix dimensions incompatible: $(size(rhs1)) × $(size(rhs2))"))
    T = __my_promote_type(A, B)
    out = cuNumeric.zeros(T, (size(rhs1, 1), size(rhs2, 2)))
    return nda_three_dot_arg(unchecked_promote_arr(rhs1, T), unchecked_promote_arr(rhs2, T), out)
end

function Base.:(*)(rhs1::NDArray{Bool, 2}, rhs2::NDArray{Bool, 2})
    throw(ArgumentError("cuNumeric.jl does not support matrix multiplication of two Boolean arrays"))
end

function Base.:(*)(rhs1::NDArray{<:Integer, 2}, rhs2::NDArray{<:Integer, 2})
    #* this is a stupid.....
    throw(ArgumentError("cuNumeric.jl does not support matrix multiplication of two Integer arrays"))
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
function LinearAlgebra.mul!(out::NDArray{T, 2}, rhs1::NDArray{A, 2}, rhs2::NDArray{B, 2}) where {T <: SUPPORTED_NUMERIC_TYPES, A, B}
    #! This will probably need more checks once we support Complex number
    size(rhs1, 2) == size(rhs2, 1) || throw(DimensionMismatch("Matrix dimensions incompatible: $(size(rhs1)) × $(size(rhs2))"))
    (size(out, 1) == size(rhs1, 1) && size(out, 2) == size(rhs2, 2)) || throw(DimensionMismatch(
        "mul! output is $(size(out)), but inputs would produce $(size(rhs1,1))×$(size(rhs2,2))"
    ))
    T_OUT = __my_promote_type(A, B)
    ((T_OUT <: AbstractFloat) && (T <: Integer)) && throw(ArgumentError("mul! output has integer type $(T), but inputs promote to floating point type: $(T_OUT)"))
    return nda_three_dot_arg(checked_promote_arr(rhs1, T), checked_promote_arr(rhs2, T), out)
end

function LinearAlgebra.mul!(out::NDArray, rhs1::NDArray{Bool, 2}, rhs2::NDArray{Bool, 2})
    #* Could just promote both inputs to Int32
    throw(ArgumentError("cuNumeric.jl does not support matrix multiplication of two Boolean arrays"))
end

function LinearAlgebra.mul!(out::NDArray, rhs1::NDArray{<:Integer, 2}, rhs2::NDArray{<:Integer, 2})
    #* this is a stupid.....
    throw(ArgumentError("cuNumeric.jl does not support matrix multiplication of two Integer arrays"))
end

# Generate hidden broadcast functions for binary ops
for (julia_fn, op_code) in broadcasted_binary_op_map
    @eval begin
        @inline function __broadcast(f::typeof($(julia_fn)), out::NDArray, rhs1::NDArray{T}, rhs2::NDArray{T}) where T
            return nda_binary_op(out, $(op_code), rhs1, rhs2)
        end
    end
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
@inline function __broadcast(f::typeof(Base.literal_pow), out::NDArray, _, input::NDArray{T}, power::NDArray{T}) where {T <: SUPPORTED_TYPES}
    return nda_binary_op(out, cuNumeric.POWER, input, power)
end


# This is more "Julian" since a user expects map to broadcast
# their operation whereas the generated functions should technically
# only broadcast when the .() syntax is used
function Base.map(f::Function, arr1::NDArray, arr2::NDArray)
    return f.(arr1, arr2) # Will try to call one of the functions generated above
end

# function Base.map!(f::Function, dest::NDArray, arr1::NDArray, arr2::NDArray)
#     return f
# end
