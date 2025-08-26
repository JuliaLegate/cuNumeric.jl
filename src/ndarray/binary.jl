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
    # Base.:(==) => (cuNumeric.EQUAL, #*  BE SURE TO DEFINE NON-BROADCASTED VERSION (BINARY_REDUCTION)
)


# @enum OUTPUT_RULES same_size_float same_size_int same_as_input

# # Functions which support only a subset of the supported types as input
# # Last value in the tuple is the return type
# global const binary_op_specific_input = Dict{Function, Tuple{BinaryOpCode, Type, Symbol}}(
#     Base.atan => (cuNumeric.ARCTAN2, SUPPORTED_NUMERIC_TYPES, :same_size_float), #technically Julia promotes Int32 inputs to FP64
#     Base.lcm => (cuNumeric.LCM, SUPPORTED_INT_TYPES, :same_as_input), 
#     Base.gcd => (cuNumeric.GCD, SUPPORTED_INT_TYPES, :same_as_input), 
#     # Base.:(&&) => (cuNumeric.LOGICAL_AND, Bool, :same_as_input), #! CANNOT OVERLOAD WTF?
#     # Base.:(||) => (cuNumeric.LOGICAL_OR, Bool, :same_as_input), #! CANNOT OVERLOAD WTF?
#     Base.xor  => (cuNumeric.LOGICAL_XOR, Bool, :same_as_input), 
#     Base.:⊻ => (cuNumeric.LOGICAL_XOR, Bool, :same_as_input),
#     Base.div => (cuNumeric.FLOOR_DIVIDE, SUPPORTED_NUMERIC_TYPES, :same_size_int),
#     Base.:(÷) => (cuNumeric.FLOOR_DIVIDE, SUPPORTED_NUMERIC_TYPES, :same_size_int),
#     # Base.:(>>) => (cuNumeric.RIGHT_SHIFT, Union{SUPPORTED_INT_TYPES, Bool}, :same_size_float) # bool input --> Int64 output in Julia
#     # Base.:(<<) => (cuNumeric.LEFT_SHIFT, Union{SUPPORTED_INT_TYPES, Bool}, :same_size_float) # bool input --> Int64 output in Julia
# )


## SPECIAL CASES ##

#! WHY CAN I NOT CALL THIS??
function Base.:(*)(rhs1::NDArray{A, 2}, rhs2::NDArray{B, 2}) where {A <: SUPPORTED_TYPES, B <: SUPPORTED_TYPES}
    T = __my_promote_type(A, B)
    out = cuNumeric.zeros(T, (size(rhs1, 1), size(rhs2, 2)))
    return nda_three_dot_arg(maybe_promote_arr(rhs1, T), maybe_promote_arr(rhs2, T), out)
end

function Base.:(*)(rhs1::NDArray{A, 2}, rhs2::NDArray{A, 2}) where A
    T = __my_promote_type(A, B)
    out = cuNumeric.zeros(T, (size(rhs1, 1), size(rhs2, 2)))
    return nda_three_dot_arg(maybe_promote_arr(rhs1, T), maybe_promote_arr(rhs2, T), out)
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

#! WHY CAN I NOT CALL THIS??
#! will probably crash horribly if input is Floats and output is Ints
function LinearAlgebra.mul!(out::NDArray{T, 2}, rhs1::NDArray{A, 2}, rhs2::NDArray{B, 2}) where {T, A, B}
    return nda_three_dot_arg(maybe_promote_arr(rhs1, T), maybe_promote_arr(rhs2, T), out)
end

# Generate hidden broadcast functions for binary ops
for (julia_fn, op_code) in broadcasted_binary_op_map
    @eval begin
        @inline function __broadcast(f::typeof($(julia_fn)), out::NDArray, rhs1::NDArray{T}, rhs2::NDArray{T}) where {T <: SUPPORTED_TYPES}
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
