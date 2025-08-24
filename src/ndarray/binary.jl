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
# global const binary_op_map = Dict{Function,BinaryOpCode}(
#     Base.:+ => cuNumeric.ADD,
#     # Base.copysign => cuNumeric.COPYSIGN, #* ANNOYING TO TEST 
#     Base.:/ => cuNumeric.DIVIDE,
#     # Base.:^ => cuNumeric.FLOAT_POWER, # DONT THINK THIS IS WHAT WE WANT
#     # Base.:^ => cuNumeric.POWER, #* HOW TO FIGURE OUT RETURN TYPE???
#     #missing => cuNumeric.fmod, #same as mod in Julia?
#     Base.hypot => cuNumeric.HYPOT,
#     # Base.isapprox => cuNumeric.ISCLOSE, #* HANDLE rtol, atol kwargs!!!
#     # Base.ldexp => cuNumeric.LDEXP, #* LHS FLOATS, RHS INTS
#     #missing => cuNumeric.LOGADDEXP,
#     #missing => cuNumeric.LOGADDEXP2,
#     #missing => cuNumeric.MAXIMUM, #elementwise max?
#     #missing => cuNumeric.MINIMUM, #elementwise min?
#     Base.:* => cuNumeric.MULTIPLY, #elementwise product? == .* in Julia
#     #missing => cuNumeric.NEXTAFTER,
#     Base.:(-) => cuNumeric.SUBTRACT)

global const binary_op_map = Dict{Function,Tuple{BinaryOpCode, Symbol}}(
    Base.:+ => (cuNumeric.ADD, :__binop_elwise_add),
    Base.:* => (cuNumeric.MULTIPLY, :__binop_elwise_mul)
)

# # Functions which allow any of the supported types as input
# # Last value in tuple is the return type
# global const binary_op_specific_return = Dict{Function, Tuple{BinaryOpCode, DataType}}(
#     Base.:(<) => (cuNumeric.LESS, Bool), #* ANNOYING TO TEST (no == for bools
#     Base.:(<=) => (cuNumeric.LESS_EQUAL, Bool),  #* ANNOYING TO TEST (no == for bools
#     Base.:(>) => (cuNumeric.GREATER, Bool), #* ANNOYING TO TEST (no == for bools
#     Base.:(>=) => (cuNumeric.GREATER_EQUAL, Bool), #* ANNOYING TO TEST (no == for bools
#     # Base.:(!=) => (cuNumeric.NOT_EQUAL, Bool), #* DONT REALLY WANT ELEMENTWISE !=, RATHER HAVE REDUCTION
#     # Base.:(==) => (cuNumeric.EQUAL, Bool),  #* This is elementwise .==, but non-broadcasted this is array_equal
# )

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


# Generate hidden functions for all binary operations.
for (_, (op_code, hidden_name)) in binary_op_map
    # Definitions and type promotion rules
    @eval begin

        @inline function $(Symbol(hidden_name))(rhs1::NDArray{T}, rhs2::NDArray{T}) where {T <: SUPPORTED_TYPES}
            out = cuNumeric.zeros(T, size(rhs1)) # wrap cupynumeric broadcast_shape function??
            return nda_binary_op(out, $(op_code), rhs1, rhs2)
        end
        
        # With same types, no promotion
        # @inline function $(Symbol(base_func))(rhs1::NDArray{T}, rhs2::NDArray{T}) where {T <: SUPPORTED_TYPES}
        #     out = cuNumeric.zeros(T, promote_shape(size(rhs1), size(rhs2)))
        #     return nda_binary_op(out, $(op_code), rhs1, rhs2)
        # end

        # # # With un-matched types, promote to same type and call back to other function
        # @inline function $(Symbol(base_func))(rhs1::NDArray{A}, rhs2::NDArray{B}) where {A <: SUPPORTED_TYPES, B <: SUPPORTED_TYPES} 
        #     T = __my_promote_type(A, B)
        #     return  $(Symbol(base_func))(maybe_promote_arr(rhs1, T), maybe_promote_arr(rhs2, T))
        # end

        # @inline function $(Symbol(base_func))(arr::NDArray{T}, c::T) where T
        #     return $(Symbol(base_func))(arr, NDArray(c))
        # end
        
        # @inline function $(Symbol(base_func))(c::T, arr::NDArray{T}) where T
        #     return $(Symbol(base_func))(NDArray(c), arr)
        # end

        # @inline function $(Symbol(base_func))(c::A, arr::NDArray{B}) where {A <: Number, B <: Number}
        #     T = __my_promote_type(A, B)
        #     return $(Symbol(base_func))(NDArray(T(c)), maybe_promote_arr(arr, T))
        # end

        # @inline function $(Symbol(base_func))(arr::NDArray{B}, c::A) where {A <: Number, B <: Number}
        #     T = __my_promote_type(A, B)
        #     return $(Symbol(base_func))(maybe_promote_arr(arr, T), NDArray(T(c)))
        # end

    end
end

# This is more "Julian" since a user expects map to broadcast
# their operation whereas the generated functions should technically
# only broadcast when the .() syntax is used
function Base.map(f::Function, arr1::NDArray, arr2::NDArray)
    return f(arr1, arr2) # Will try to call one of the functions generated above
end

# function Base.map!(f::Function, dest::NDArray, arr1::NDArray, arr2::NDArray)
#     return f
# end
