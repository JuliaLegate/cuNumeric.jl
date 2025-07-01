export square

global const unary_op_map_no_args = Dict{Union{Function, Symbol}, Int}(
    Base.abs => Int(cuNumeric.ABSOLUTE),
    Base.acos => Int(cuNumeric.ARCCOS), 
    # Base.acosh => Int(cuNumeric.ARCCOSH), #* makes testing annoying
    Base.asin => Int(cuNumeric.ARCSIN),
    Base.asinh => Int(cuNumeric.ARCSINH),
    Base.atan => Int(cuNumeric.ARCTAN),
    Base.atanh => Int(cuNumeric.ARCTANH),
    Base.cbrt => Int(cuNumeric.CBRT),
    Base.conj => Int(cuNumeric.CONJ),
    # missing => Int(cuNumeric.COPY), # SAME AS ASSIGN DONT NEED, OR COULD HARD CODE TO USE
    Base.cos => Int(cuNumeric.COS),
    Base.cosh => Int(cuNumeric.COSH),
    Base.deg2rad => Int(cuNumeric.DEG2RAD),
    Base.exp => Int(cuNumeric.EXP),
    Base.exp2 => Int(cuNumeric.EXP2),
    Base.expm1 => Int(cuNumeric.EXPM1),
    Base.floor => Int(cuNumeric.FLOOR),
    # Base.frexp => Int(cuNumeric.FREXP), #* makes testing annoying
    #missing => Int(cuNumeric.GETARG), #not in numpy?
    # Base.imag => Int(cuNumeric.IMAG), #* makes testing annoying
    #missing => Int(cuNumerit.INVERT), # 1/x or inv(A)?
    # Base.isfinite => Int(cuNumeric.ISFINITE), #* makes testing annoying
    # Base.isinf => Int(cuNumeric.ISINF), #* makes testing annoying
    # Base.isnan => Int(cuNumeric.ISNAN), #* makes testing annoying
    Base.log => Int(cuNumeric.LOG),
    Base.log10 => Int(cuNumeric.LOG10),
    Base.log1p => Int(cuNumeric.LOG1P),
    Base.log2 => Int(cuNumeric.LOG2),
    # Base.:! => Int(cuNumeric.LOGICAL_NOT), #* makes testing annoying
    # Base.modf => Int(cuNumeric.MODF), #* makes testing annoying
    Base.:- => Int(cuNumeric.NEGATIVE), 
    #missing => Int(cuNumeric.POSITIVE), #What is this even for
    Base.rad2deg => Int(cuNumeric.RAD2DEG),
    # Base.sign => Int(cuNumeric.SIGN), #* makes testing annoying
    # Base.signbit => Int(cuNumeric.SIGNBIT), #* makes testing annoying
    Base.sin => Int(cuNumeric.SIN),  
    Base.sinh => Int(cuNumeric.SINH),  
    Base.sqrt => Int(cuNumeric.SQRT),  # HAS SPECIAL MEANING FOR MATRIX
    :square => Int(cuNumeric.SQUARE),
    Base.tan => Int(cuNumeric.TAN),  
    Base.tanh => Int(cuNumeric.TANH),  
)

"""
square(arr::NDArray)

Elementwise square of each element in `arr`. 
"""
function square end


# Generate code for all unary operators
for (base_func, op_code) in unary_op_map_no_args
    @eval begin
        function $(Symbol(base_func))(input::NDArray)
            out = cuNumeric.zeros(eltype(input), Base.size(input)) # not sure this is ok for performance
            # empty = Legate.VectorScalar() # not sure this is ok for performanc
            return nda_unary_op(out, $(op_code), input)
        end
    end
end

# global const unary_op_map_with_args = Dict{Function, Int}(
#     Base.angle => Int(cuNumeric.ANGLE),
#     Base.ceil => Int(cuNumeric.CEIL), #* HAS EXTRA ARGS
#     Base.clamp => Int(cuNumeric.CLIP), #* HAS EXTRA ARGS
#     Base.trunc => Int(cuNumeric.TRUNC)  #* HAS EXTRA ARGS
#     missing => Int(cuNumeric.RINT), #figure out which version of round 
#     missing => Int(cuNumeric.ROUND), #figure out which version of round
# )

# for (base_func, op_code) in unary_op_map_with_args
#     @eval begin
#         @doc """
#             $($(Symbol(base_func))) : A unary operation acting on an NDArray
#         """
#         function $(Symbol(base_func))(input::NDArray, args...)
#             out = cuNumeric.zeros(eltype(input), size(input)) # not sure this is ok for performance
#             extra_args = cuNumeric.StdVector{cuNumeric.LegateScalar}([LegateScalar(a) for a in args])
#             unary_op(out, $(op_code), input, extra_args)
#             return out
#         end
#     end
# end

# Could implement most of the missing functions here
global const unary_reduction_map = Dict{Function, Int}(
    # Base.all => Int(cuNumeric.ALL), #* ANNOYING TO TEST
    # Base.any => Int(cuNumeric.ANY), #* ANNOYING TO TEST
    # Base.argmax => Int(cuNumeric.ARGMAX), #* WILL BE OFF BY 1
    # Base.argmin => Int(cuNumeric.ARGMIN), #* WILL BE OFF BY 1
    #missing => Int(cuNumeric.CONTAINS), # strings or also integral types
    #missing => Int(cuNumeric.COUNT_NONZERO),
    Base.maximum => Int(cuNumeric.MAX),
    Base.minimum => Int(cuNumeric.MIN),
    #missing => Int(cuNumeric.NANARGMAX),
    #missing => Int(cuNumeric.NANARGMIN),
    #missing => Int(cuNumeric.NANMAX),
    #missing => Int(cuNumeric.NANMIN),
    #missing => Int(cuNumeric.NANPROD),
    Base.prod => Int(cuNumeric.PROD),
    Base.sum => Int(cuNumeric.SUM),
    #missing => Int(cuNumeric.SUM_SQUARES),
    #missing => Int(cuNumeric.VARIANCE)
)



# #*TODO HOW TO GET THESE ACTING ON CERTAIN DIMS
# Generate code for all unary reductions.
for (base_func, op_code) in unary_reduction_map
    @eval begin
        function $(Symbol(base_func))(input::NDArray)
            #* WILL BREAK NOT ALL REDUCTIONS HAVE SAME TYPE AS INPUT
            out = cuNumeric.zeros(eltype(input), 1) # not sure this is ok for performance
            return nda_unary_reduction(out, $(op_code), input)
        end
    end
end

# function Base.reduce(f::Function, arr::NDArray)
#     return f(arr)
# end


#### PROVIDE A MORE "JULIAN" WAY OF DOING THINGS
#### WHEN YOU CALL MAP YOU EXPECT BROADCASTING
#### THIS HAS SOME EXTRA OVERHEAD THOUGH SINCE
#### YOU HAVE TO LOOK UP THE OP CODE AND CHECK IF ITS VALID


#* TODO Overload broadcasting to just call this
#* e.g. sin.(ndarray) should call this or the proper generated func
function Base.map(f::Function, arr::NDArray)
    return f(arr) # Will try to call one of the functions generated above
end


# function get_unary_op(f::Function)
#     if haskey(unary_op_map, f)
#         return unary_op_map[f]
#     else
#         throw(KeyError("Unsupported unary operation : $(f)"))
#     end
# end

# function Base.map(f::Function, arr::NDArray)
#     out = cuNumeric.zeros(eltype(arr), size(arr)) # not sure this is ok for performance
#     op_code = get_unary_op(f)
#     return unary_op(out, op_code, arr)
# end