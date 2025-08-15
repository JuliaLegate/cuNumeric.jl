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
global const binary_op_map = Dict{Function,BinaryOpCode}(
    Base.:+ => cuNumeric.ADD,
    # Base.copysign => cuNumeric.COPYSIGN, #* ANNOYING TO TEST 
    Base.:/ => cuNumeric.DIVIDE,
    # Base.:^ => cuNumeric.FLOAT_POWER, # DONT THINK THIS IS WHAT WE WANT
    # Base.:^ => cuNumeric.POWER, #* HOW TO FIGURE OUT RETURN TYPE???
    #missing => cuNumeric.fmod, #same as mod in Julia?
    Base.hypot => cuNumeric.HYPOT,
    # Base.isapprox => cuNumeric.ISCLOSE, #* HANDLE rtol, atol kwargs!!!
    # Base.ldexp => cuNumeric.LDEXP, #* LHS FLOATS, RHS INTS
    #missing => cuNumeric.LOGADDEXP,
    #missing => cuNumeric.LOGADDEXP2,
    #missing => cuNumeric.MAXIMUM, #elementwise max?
    #missing => cuNumeric.MINIMUM, #elementwise min?
    Base.:* => cuNumeric.MULTIPLY, #elementwise product? == .* in Julia
    #missing => cuNumeric.NEXTAFTER,
    Base.:(-) => cuNumeric.SUBTRACT)



# Functions which allow any of the supported types as input
# Last value in tuple is the return type
global const binary_op_specific_return = Dict{Function, Tuple{BinaryOpCode, DataType}}(
    Base.:(<) => (cuNumeric.LESS, Bool), #* ANNOYING TO TEST (no == for bools
    Base.:(<=) => (cuNumeric.LESS_EQUAL, Bool),  #* ANNOYING TO TEST (no == for bools
    Base.:> => (cuNumeric.GREATER, Bool), #* ANNOYING TO TEST (no == for bools
    Base.:(>=) => (cuNumeric.GREATER_EQUAL, Bool), #* ANNOYING TO TEST (no == for bools
    # Base.:(!=) => (cuNumeric.NOT_EQUAL, Bool), #* DONT REALLY WANT ELEMENTWISE !=, RATHER HAVE REDUCTION
    # Base.:(==) => (cuNumeric.EQUAL, Bool),  #* DONT REALLY WANT ELEMENTWISE ==, RATHER HAVE REDUCTION
)

# Functions which support only a subset of the supported types as input
# Last value in the tuple is the return type
global const binary_op_specific_input = Dict{Function, Tuple{BinaryOpCode, DataType, Symbol}}(
    Base.atan => (cuNumeric.ARCTAN2, SUPPORTED_NUMERIC_TYPES, :same_size_float), #technically Julia promotes Int32 inputs to FP64
    Base.lcm => (cuNumeric.LCM, SUPPORTED_INT_TYPES, :same_as_input), 
    Base.gcd => (cuNumeric.GCD, SUPPORTED_INT_TYPES, :same_as_input), 
    Base.:&& => (cuNumeric.LOGICAL_AND, Bool, :Bool),
    Base.:|| => (cuNumeric.LOGICAL_OR, Bool, :Bool),
    Base.xor  => (cuNumeric.LOGICAL_XOR, Bool, :Bool), 
    Base.:⊻ => (cuNumeric.LOGICAL_XOR, Bool, :Bool),
    Base.div => (cuNumeric.FLOOR_DIVIDE, SUPPORTED_NUMERIC_TYPES, :same_size_int),
    Base.:(÷) => (cuNumeric.FLOOR_DIVIDE, SUPPORTED_NUMERIC_TYPES, :same_size_int),
    # Base.:(>>) => (cuNumeric.RIGHT_SHIFT, Union{SUPPORTED_INT_TYPES, Bool}, :same_size_float) # bool input --> Int64 output in Julia
    # Base.:(<<) => (cuNumeric.LEFT_SHIFT, Union{SUPPORTED_INT_TYPES, Bool}, :same_size_float) # bool input --> Int64 output in Julia
)


maybe_promote_arr(arr::NDArray{T}, ::Type{T}) where T = arr
maybe_promote_arr(arr::NDArray{T}, ::Type{S}) where {T,S} = as_type(arr, S)

smaller_type(::Type{A}, ::Type{B}) where {A,B} = ifelse(sizeof(A) < sizeof(B), A, B)
same_size(::Type{A}, ::Type{B}) where {A,B} = sizeof(A) == sizeof(B)

function __my_promote_type(::Type{A}, ::Type{B}) where {A, B}
    T = promote_type(A, B)
    same_size(A, B) && return T
    S = smaller_type(A, B)
    S != T && error("Detected promotion from $S to larger type, $T")
    return T
end

#* THIS SORT OF BREAKS WHAT A JULIA USER MIGHT EXPECT
#* WILL AUTOMATICALLY BROADCAST OVER ARRAY INSTEAD OF REQUIRING `.()` call sytax
#* NEED TO IMPLEMENT BROADCASTING INTERFACE
# Generate code for all binary operations.
for (base_func, op_code) in binary_op_map
    # Definitions and type promotion rules
    @eval begin
        
        # With same types, no promotion
        @inline function $(Symbol(base_func))(rhs1::NDArray{T}, rhs2::NDArray{T}) where {T <: Number}
            out = cuNumeric.zeros(T, Base.size(rhs1))
            return nda_binary_op(out, $(op_code), rhs1, rhs2)
        end

        # # With un-matched types, promote to same type and call back to other function
        @inline function $(Symbol(base_func))(rhs1::NDArray{A}, rhs2::NDArray{B}) where {A <: Number, B <: Number} 
            T = __my_promote_type(A, B)
            return  $(Symbol(base_func))(maybe_promote_arr(rhs1, T), maybe_promote_arr(rhs2, T))
        end

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
