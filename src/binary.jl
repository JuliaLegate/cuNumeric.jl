@doc"""
Supported Binary Operations
===========================

The following binary operations are supported and can be applied elementwise to pairs of `NDArray` values:

  • `+`        — Addition  
  • `-`        — Subtraction  
  • `*`        — Multiplication (elementwise)  
  • `/`        — Division (elementwise)  
  • `^`        — Floating-point power (elementwise)  
  • `div`      — Floor division  
  • `atan`     — Two-argument arctangent (`atan(y, x)`)  
  • `hypot`    — Hypotenuse (`hypot(x, y)`)

These operations are broadcast-compatible and follow Julia's standard operator overloading behavior.

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
    Base.:^ => cuNumeric.FLOAT_POWER, # diff from POWER?
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
    #Base.:^ => cuNumeric.POWER,
    # Base.:(>>) => cuNumeric.RIGHT_SHIFT, #* ANNOYING TO TEST (no == for bools)
    Base.:(-) => cuNumeric.SUBTRACT)

#* THIS SORT OF BREAKS WHAT A JULIA USER MIGHT EXPECT
#* WILL AUTOMATICALLY BROADCAST OVER ARRAY INSTEAD OF REQUIRING `.()` call sytax
#* NEED TO IMPLEMENT BROADCASTING INTERFACE
# Generate code for all binary operations.
for (base_func, op_code) in binary_op_map
    @eval begin
        function $(Symbol(base_func))(rhs1::NDArray, rhs2::NDArray)
            #* what happens if rhs1 and rhs2 have different types but are compatible?
            out = cuNumeric.zeros(cuNumeric.eltype(rhs1), Base.size(rhs1)) # not sure this is ok for performance
            return nda_binary_op(out, $(op_code), rhs1, rhs2)
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
