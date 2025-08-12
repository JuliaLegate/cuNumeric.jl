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
    Base.atan => cuNumeric.ARCTAN2,
    # Base.:& => cuNumeric.BITWISE_AND, #* ANNOYING TO TEST (no == for bools)
    # Base.:| => cuNumeric.BITWISE_OR, #* ANNOYING TO TEST (no == for bools)
    # Base.:⊻ => cuNumeric.BITWISE_XOR, #* ANNOYING TO TEST (no == for bools)
    # Base.copysign => cuNumeric.COPYSIGN, #* ANNOYING TO TEST 
    Base.:/ => cuNumeric.DIVIDE,
    # Base.:(==) => cuNumeric.EQUAL,  #* DONT REALLY WANT ELEMENTWISE ==, RATHER HAVE REDUCTION
    # Base.:^ => cuNumeric.FLOAT_POWER, # DONT THINK THIS IS WHAT WE WANT
    Base.:^ => cuNumeric.POWER,
    Base.div => cuNumeric.FLOOR_DIVIDE,
    #missing => cuNumeric.fmod, #same as mod in Julia?
    # Base.gcd => cuNumeric.GCD, #* ANNOYING TO TEST (need ints)
    # Base.:> => cuNumeric.GREATER, #* ANNOYING TO TEST (no == for bools
    # Base.:(>=) => cuNumeric.GREATER_EQUAL, #* ANNOYING TO TEST (no == for bools
    Base.hypot => cuNumeric.HYPOT,
    # Base.isapprox => cuNumeric.ISCLOSE, #* ANNOYING TO TEST (no == for bools
    # Base.lcm => cuNumeric.LCM,  #* ANNOYING TO TEST (need ints)
    # Base.ldexp => cuNumeric.LDEXP, #* ANNOYING TO TEST (need ints)
    # Base.:(<<) => cuNumeric.LEFT_SHIFT,  #* ANNOYING TO TEST (no == for bools)
    # Base.:(<) => cuNumeric.LESS, #* ANNOYING TO TEST (no == for bools
    # Base.:(<=) => cuNumeric.LESS_EQUAL,  #* ANNOYING TO TEST (no == for bools
    #missing => cuNumeric.LOGADDEXP,
    #missing => cuNumeric.LOGADDEXP2,
    # Base.:&& => cuNumeric.LOGICAL_AND, # This returns bits?
    # Base.:|| => cuNumeric.LOGICAL_OR, #This returns bits?
    #missing  => cuNumeric.LOGICAL_XOR,
    #missing => cuNumeric.MAXIMUM, #elementwise max?
    #missing => cuNumeric.MINIMUM, #elementwise min?
    Base.:* => cuNumeric.MULTIPLY, #elementwise product? == .* in Julia
    #missing => cuNumeric.NEXTAFTER,
    # Base.:(!=) => cuNumeric.NOT_EQUAL, #* DONT REALLY WANT ELEMENTWISE !=, RATHER HAVE REDUCTION
    # Base.:(>>) => cuNumeric.RIGHT_SHIFT, #* ANNOYING TO TEST (no == for bools)
    Base.:(-) => cuNumeric.SUBTRACT)


maybe_promote_arr(arr::NDArray{T}, ::Type{T}) where T = arr
maybe_promote_arr(arr::NDArray{T}, ::Type{S}) where {T,S} = as_type(arr, S)

smaller_type(::A, ::B) where {A,B} = ifelse(sizeof(A) < sizeof(B), A, B)

function __my_promote_type(x::Type, y::Type)
    S = smaller_type(x, y)
    T = promote_type(x, y)
    S != T || error("Detected promotion from $S to larger type, $T")
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
        @inline function $(Symbol(base_func))(rhs1::NDArray{T}, rhs2::NDArray{T}) where T
            out = cuNumeric.zeros(T, Base.size(rhs1))
            return nda_binary_op(out, $(op_code), rhs1, rhs2)
        end

        # With un-matched types, promote to same type and call back to other function
        @inline function $(Symbol(base_func))(rhs1::NDArray{A}, rhs2::NDArray{B}) where {A <: Number, B <: Number} 
            T = __my_promote_type(A, B)
            return  $(Symbol(base_func))(maybe_promote_arr(rhs1, T), maybe_promote_arr(rhs2, T))
        end

        #! Need to add support in C++ for this!!!!
        @inline function $(Symbol(base_func))(arr::NDArray{T}, c::T) where T
            error("Not yet implemented")
            # return $(Symbol(base_func))(T(c), maybe_promote_arr(arr, T))
        end
        
        @inline function $(Symbol(base_func))(c::T, arr::NDArray{T}) where T
            error("Not yet implemented")
            # return $(Symbol(base_func))(arr, c)
        end

        @inline function $(Symbol(base_func))(c::A, arr::NDArray{B}) where {A <: Number, B <: Number}
            error("Not yet implemented")
            # T = __my_promote_type(A, B)
            # return $(Symbol(base_func))(T(c), maybe_promote_arr(arr, T))
        end

        @inline function $(Symbol(base_func))(arr::NDArray{B}, c::A) where {A <: Number, B <: Number}
            error("Not yet implemented")
            # T = __my_promote_type(A, B)
            # return $(Symbol(base_func))(T(c), maybe_promote_arr(arr, T))
        end

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
