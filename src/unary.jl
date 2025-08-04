export square

@doc"""
Supported Unary Operations
===========================

The following unary operations are supported and can be applied directly to `NDArray` values:

  • `abs`
  • `acos`
  • `asin`
  • `asinh`
  • `atan`
  • `atanh`
  • `cbrt`
  • `conj`
  • `cos`
  • `cosh`
  • `deg2rad`
  • `exp`
  • `exp2`
  • `expm1`
  • `floor`
  • `log`
  • `log10`
  • `log1p`
  • `log2`
  • `-` (negation)
  • `rad2deg`
  • `sin`
  • `sinh`
  • `sqrt`
  • `square`
  • `tan`
  • `tanh`

These operations are applied elementwise by default and follow standard Julia semantics.

Examples
--------

```julia
A = NDArray(randn(Float32, 3, 3))

abs(A)
log.(A .+ 1)
-sqrt(abs(A))
square(A)
```
"""
global const unary_op_map_no_args = Dict{Union{Function,Symbol},UnaryOpCode}(
    Base.abs => cuNumeric.ABSOLUTE,
    Base.acos => cuNumeric.ARCCOS,
    # Base.acosh => cuNumeric.ARCCOSH, #* makes testing annoying
    Base.asin => cuNumeric.ARCSIN,
    Base.asinh => cuNumeric.ARCSINH,
    Base.atan => cuNumeric.ARCTAN,
    Base.atanh => cuNumeric.ARCTANH,
    Base.cbrt => cuNumeric.CBRT,
    Base.conj => cuNumeric.CONJ,
    # missing => cuNumeric.COPY, # SAME AS ASSIGN DONT NEED, OR COULD HARD CODE TO USE
    Base.cos => cuNumeric.COS,
    Base.cosh => cuNumeric.COSH,
    Base.deg2rad => cuNumeric.DEG2RAD,
    Base.exp => cuNumeric.EXP,
    Base.exp2 => cuNumeric.EXP2,
    Base.expm1 => cuNumeric.EXPM1,
    Base.floor => cuNumeric.FLOOR,
    # Base.frexp => cuNumeric.FREXP, #* makes testing annoying
    #missing => cuNumeric.GETARG, #not in numpy?
    # Base.imag => cuNumeric.IMAG, #* makes testing annoying
    #missing => cuNumerit.INVERT, # 1/x or inv(A)?
    # Base.isfinite => cuNumeric.ISFINITE, #* makes testing annoying
    # Base.isinf => cuNumeric.ISINF, #* makes testing annoying
    # Base.isnan => cuNumeric.ISNAN, #* makes testing annoying
    Base.log => cuNumeric.LOG,
    Base.log10 => cuNumeric.LOG10,
    Base.log1p => cuNumeric.LOG1P,
    Base.log2 => cuNumeric.LOG2,
    # Base.:! => cuNumeric.LOGICAL_NOT, #* makes testing annoying
    # Base.modf => cuNumeric.MODF, #* makes testing annoying
    Base.:- => cuNumeric.NEGATIVE,
    #missing => cuNumeric.POSITIVE, #What is this even for
    Base.rad2deg => cuNumeric.RAD2DEG,
    # Base.sign => cuNumeric.SIGN, #* makes testing annoying
    # Base.signbit => cuNumeric.SIGNBIT, #* makes testing annoying
    Base.sin => cuNumeric.SIN,
    Base.sinh => cuNumeric.SINH,
    Base.sqrt => cuNumeric.SQRT,  # HAS SPECIAL MEANING FOR MATRIX
    :square => cuNumeric.SQUARE,
    Base.tan => cuNumeric.TAN,
    Base.tanh => cuNumeric.TANH,
)

@doc"""
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

@doc"""
Supported Unary Reduction Operations
===========================

The following unary reduction operations are supported and can be applied directly to `NDArray` values:

  • `maximum`
  • `minimum`
  • `prod`
  • `sum`


These operations follow standard Julia semantics.

Examples
--------

```julia
A = NDArray(randn(Float32, 3, 3))

maximum(A)
sum(A)
```
"""
global const unary_reduction_map = Dict{Function,UnaryRedCode}(
    # Base.all => cuNumeric.ALL, #* ANNOYING TO TEST
    # Base.any => cuNumeric.ANY, #* ANNOYING TO TEST
    # Base.argmax => cuNumeric.ARGMAX, #* WILL BE OFF BY 1
    # Base.argmin => cuNumeric.ARGMIN, #* WILL BE OFF BY 1
    #missing => cuNumeric.CONTAINS, # strings or also integral types
    #missing => cuNumeric.COUNT_NONZERO,
    Base.maximum => cuNumeric.MAX,
    Base.minimum => cuNumeric.MIN,
    #missing => cuNumeric.NANARGMAX,
    #missing => cuNumeric.NANARGMIN,
    #missing => cuNumeric.NANMAX,
    #missing => cuNumeric.NANMIN,
    #missing => cuNumeric.NANPROD,
    Base.prod => cuNumeric.PROD,
    Base.sum => cuNumeric.SUM,
    #missing => cuNumeric.SUM_SQUARES,
    #missing => cuNumeric.VARIANCE
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
