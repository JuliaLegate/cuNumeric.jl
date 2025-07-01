global const binary_op_map = Dict{Function, Int}(
    Base.:+ => Int(cuNumeric.ADD),
    Base.atan => Int(cuNumeric.ARCTAN2),
    # Base.:& => Int(cuNumeric.BITWISE_AND), #* ANNOYING TO TEST (no == for bools)
    # Base.:| => Int(cuNumeric.BITWISE_OR), #* ANNOYING TO TEST (no == for bools)
    # Base.:⊻ => Int(cuNumeric.BITWISE_XOR), #* ANNOYING TO TEST (no == for bools)
    # Base.copysign => Int(cuNumeric.COPYSIGN), #* ANNOYING TO TEST 
    Base.:/ => Int(cuNumeric.DIVIDE),
    # Base.:(==) => Int(cuNumeric.EQUAL),  #* DONT REALLY WANT ELEMENTWISE ==, RATHER HAVE REDUCTION
    Base.:^ => Int(cuNumeric.FLOAT_POWER), # diff from POWER?
    Base.div => Int(cuNumeric.FLOOR_DIVIDE),
    #missing => Int(cuNumeric.fmod), #same as mod in Julia?
    # Base.gcd => Int(cuNumeric.GCD), #* ANNOYING TO TEST (need ints)
    # Base.:> => Int(cuNumeric.GREATER), #* ANNOYING TO TEST (no == for bools
    # Base.:(>=) => Int(cuNumeric.GREATER_EQUAL), #* ANNOYING TO TEST (no == for bools
    Base.hypot => Int(cuNumeric.HYPOT),
    # Base.isapprox => Int(cuNumeric.ISCLOSE), #* ANNOYING TO TEST (no == for bools
    # Base.lcm => Int(cuNumeric.LCM),  #* ANNOYING TO TEST (need ints)
    # Base.ldexp => Int(cuNumeric.LDEXP), #* ANNOYING TO TEST (need ints)
    # Base.:(<<) => Int(cuNumeric.LEFT_SHIFT),  #* ANNOYING TO TEST (no == for bools)
    # Base.:(<) => Int(cuNumeric.LESS), #* ANNOYING TO TEST (no == for bools
    # Base.:(<=) => Int(cuNumeric.LESS_EQUAL),  #* ANNOYING TO TEST (no == for bools
    #missing => Int(cuNumeric.LOGADDEXP),
    #missing => Int(cuNumeric.LOGADDEXP2),
    # Base.:&& => Int(cuNumeric.LOGICAL_AND), # This returns bits?
    # Base.:|| => Int(cuNumeric.LOGICAL_OR), #This returns bits?
    #missing  => Int(cuNumeric.LOGICAL_XOR),
    #missing => Int(cuNumeric.MAXIMUM), #elementwise max?
    #missing => Int(cuNumeric.MINIMUM), #elementwise min?
    Base.:* => Int(cuNumeric.MULTIPLY), #elementwise product? == .* in Julia
    #missing => Int(cuNumeric.NEXTAFTER),
    # Base.:(!=) => Int(cuNumeric.NOT_EQUAL), #* DONT REALLY WANT ELEMENTWISE !=, RATHER HAVE REDUCTION
    #Base.:^ => Int(cuNumeric.POWER),
    # Base.:(>>) => Int(cuNumeric.RIGHT_SHIFT), #* ANNOYING TO TEST (no == for bools)
    Base.:(-) => Int(cuNumeric.SUBTRACT)
    
)

#* THIS SORT OF BREAKS WHAT A JULIA USER MIGHT EXPECT
#* WILL AUTOMATICALLY BROADCAST OVER ARRAY INSTEAD OF REQUIRING `.()` call sytax
#* NEED TO IMPLEMENT BROADCASTING INTERFACE
# Generate code for all binary operations.
for (base_func, op_code) in binary_op_map
    @eval begin
        function $(Symbol(base_func))(rhs1::NDArray, rhs2::NDArray)
            #* what happens if rhs1 and rhs2 have different types but are compatible?
            out = cuNumeric.zeros(eltype(rhs1), Base.size(rhs1)) # not sure this is ok for performance
            nda_binary_op(out, $(op_code), rhs1, rhs2)
            return out
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