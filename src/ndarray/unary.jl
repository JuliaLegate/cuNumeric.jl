export square

@doc"""
Supported Unary Operations
===========================

The following unary operations are supported and can be broadcast over `NDArray`:

  - `-` (negation)
  - `!` (logical not)
  - `abs`
  - `acos`
  - `acosh`
  - `asin`
  - `asinh`
  - `atan`
  - `atanh`
  - `cbrt`
  - `cos`
  - `cosh`
  - `deg2rad`
  - `exp`
  - `exp2`
  - `expm1`
  - `floor`
  - `isfinite`
  - `log`
  - `log10`
  - `log1p`
  - `log2`
  - `rad2deg`
  - `sign`
  - `signbit`
  - `sin`
  - `sinh`
  - `sqrt`
  - `tan`
  - `tanh`
  - `^2`
  - `^-1` or `inv`

Differences
-----------
- The `acosh` function in Julia will error on inputs outside of the domain (x >= 1)
    but cuNumeric.jl will return NaN.

Examples
--------

```julia
A = cuNumeric.ones(Float32, 3, 3)

abs.(A)
log.(A .+ 1)
-sqrt.(abs.(A))
```
"""
global const floaty_unary_ops_no_args = Dict{Function, UnaryOpCode}(
    Base.acos => cuNumeric.ARCCOS,
    Base.acosh => cuNumeric.ARCCOSH,
    Base.asin => cuNumeric.ARCSIN,
    Base.asinh => cuNumeric.ARCSINH,
    Base.atan => cuNumeric.ARCTAN,
    Base.atanh => cuNumeric.ARCTANH,
    Base.cbrt => cuNumeric.CBRT,
    Base.cos => cuNumeric.COS,
    Base.cosh => cuNumeric.COSH,
    Base.deg2rad => cuNumeric.DEG2RAD,
    Base.exp => cuNumeric.EXP,
    Base.exp2 => cuNumeric.EXP2,
    Base.expm1 => cuNumeric.EXPM1,
    Base.log => cuNumeric.LOG,
    Base.log10 => cuNumeric.LOG10,
    Base.log1p => cuNumeric.LOG1P,
    Base.log2 => cuNumeric.LOG2,
    Base.rad2deg => cuNumeric.RAD2DEG,
    Base.sin => cuNumeric.SIN,
    Base.sinh => cuNumeric.SINH,
    Base.sqrt => cuNumeric.SQRT,  # HAS SPECIAL MEANING FOR MATRIX
    Base.tan => cuNumeric.TAN,
    Base.tanh => cuNumeric.TANH,
)

global const unary_op_map_no_args = Dict{Function, UnaryOpCode}(
    Base.abs => cuNumeric.ABSOLUTE,
    # Base.conj => cuNumeric.CONJ, #! NEED TO SUPPORT COMPLEX TYPES FIRST
    Base.:(-) => cuNumeric.NEGATIVE,
    # Base.frexp => cuNumeric.FREXP, #* annoying returns tuple
    # missing => cuNumeric.GETARG, #not in numpy?
    # Base.imag => cuNumeric.IMAG, #! NEED TO SUPPORT COMPLEX TYPES FIRST
    # missing => cuNumerit.INVERT, # no bitwise not in julia?
    # Base.isfinite => cuNumeric.ISFINITE, #* dont feel like looking into Inf rn
    # Base.isinf => cuNumeric.ISINF, #* dont feel like looking into Inf rn
    # Base.isnan => cuNumeric.ISNAN, #* dont feel like looking into Inf rn
    # Base.modf => cuNumeric.MODF, #* annoying returns tuple
    #missing => cuNumeric.POSITIVE, #What is this even for
    Base.sign => cuNumeric.SIGN, 
    # Base.signbit => cuNumeric.SIGNBIT, #! Doesnt support Bool, I do not feel like dealing with this right now...
)


### SPECIAL CASES ###

# Non-broadcasted version of negation
function Base.:(-)(input::NDArray{T}) where T
    out = cuNumeric.zeros(T, size(input))
    return nda_unary_op(out, cuNumeric.NEGATIVE, input)
end

function Base.:(-)(input::NDArray{Bool})
    throw(error(ArgumentError("cuNumeric.jl does not support negation (-) of Boolean NDArrays")))
end


function Base.sqrt(input::NDArray{T,2}) where T
    error("cuNumeric.jl does not support matrix square root.")
end

@inline function __broadcast(f::typeof(Base.literal_pow), out::NDArray{O}, _, input::NDArray{T}, ::Type{Val{2}}) where {T,O}
    return nda_unary_op(out, cuNumeric.SQUARE, input)
end

@inline function __broadcast(::typeof(Base.literal_pow), out::NDArray{O}, _, input::NDArray, ::Type{Val{-1}}) where O
    copyto!(out, O(1) ./ checked_promote_arr(input,O)) #! HAS EXTRA ALLOC, REPLACE WITH RECIP ONCE FIXED
    return out
    # return nda_unary_op(out, cuNumeric.RECIPROCAL, input)
end

@inline function __broadcast(::typeof(Base.inv), out::NDArray{O}, input::NDArray) where O
    copyto!(out, O(1) ./ checked_promote_arr(input,O)) #! HAS EXTRA ALLOC, REPLACE WITH RECIP ONCE FIXED
    return out
    # return nda_unary_op(out, cuNumeric.RECIPROCAL, checked_promote_arr(input,O))
end


#! NEEDS TO SUPPORT inv and ^ -1
# @inline function literal_pow(::typeof(^), A::NDArray{T, 2}, ::Val{-1}) where T
#     println("HERE")
#     #! CAN WE ADD OPTIMIZATION FOR DIAGONAL MATRIX???
#     LinearAlgebra.checksquare(A)
#     out = cuNumeric.zeros(T, size(A))
#     error("Matrix inverse not supported yet")
#     # return nda_matrix_power(out, A, -1)
# end


# Only supported for Bools
@inline function __broadcast(f::typeof(Base.:(!)), out::NDArray{Bool}, input::NDArray{Bool})
    return nda_unary_op(out, cuNumeric.LOGICAL_NOT, input)
end

# Generate hidden broadcasted version of unary ops.
for (julia_fn, op_code) in unary_op_map_no_args
    @eval begin
        @inline  function __broadcast(f::typeof($julia_fn), out::NDArray{T}, input::NDArray{T}) where T
            return nda_unary_op(out, $(op_code), input)
        end
    end
end

# Some functions always return floats even when given integers
# in the case where the output is determined to be float, but 
# the input is integer, we first promote the input to float.
for (julia_fn, op_code) in floaty_unary_ops_no_args
    @eval begin
        @inline  function __broadcast(f::typeof($julia_fn), out::NDArray{T}, input::NDArray{T}) where T
            return nda_unary_op(out, $(op_code), input)
        end

        # If input is not already float, promote to that
        @inline  function __broadcast(f::typeof($julia_fn), out::NDArray{A}, input::NDArray{B}) where {A <: SUPPORTED_FLOAT_TYPES, B <: Union{SUPPORTED_INT_TYPES, Bool}}
            return __broadcast(f, out, checked_promote_arr(input, A))
        end
    end
end

# global const unary_op_map_with_args = Dict{Function, Int}(
#     Base.angle => Int(cuNumeric.ANGLE),
#     Base.ceil => Int(cuNumeric.CEIL), #* HAS EXTRA ARGS
#     Base.clamp => Int(cuNumeric.CLIP), #* HAS EXTRA ARGS
#     Base.floor => cuNumeric.FLOOR, #! Doesnt support Bool, I do not feel like dealing with this right now...
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

  • `all`
  • `any`
  • `maximum`
  • `minimum`
  • `prod`
  • `sum`


These operations follow standard Julia semantics.

Examples
--------

```julia
A = cuNumeric.ones(5)

maximum(A)
sum(A)
```
"""
global const unary_reduction_map = Dict{Function,UnaryRedCode}(
    # Base.argmax => cuNumeric.ARGMAX, #* WILL BE OFF BY 1
    # Base.argmin => cuNumeric.ARGMIN, #* WILL BE OFF BY 1
    #missing => cuNumeric.CONTAINS, # strings or also integral types
    #missing => cuNumeric.COUNT_NONZERO, # Base.count(!Base.iszero, arr)
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
    # StatsBase.var => cuNumeric.VARIANCE #! dies horribly?? wth
)

#! IT WOULD BE NICE IF THESE JUST RETURNED SCALARS WHEN APPROPRIATE
# #*TODO HOW TO GET THESE ACTING ON CERTAIN DIMS
# Generate code for all unary reductions.
for (base_func, op_code) in unary_reduction_map
    @eval begin
        function $(Symbol(base_func))(input::NDArray{T}) where T
            T_OUT = Base.promote_op($base_func, Vector{T})
            is_wider_type(T_OUT, T) && assertpromotion($base_func, T, T_OUT)
            out = cuNumeric.zeros(T_OUT) #0D result (not right if reducing along dims)
            return nda_unary_reduction(out, $(op_code), unchecked_promote_arr(input, T_OUT))
        end
    end
end


function Base.all(input::NDArray{Bool}) 
    out = cuNumeric.zeros(Bool)
    return nda_unary_reduction(out, cuNumeric.ALL, input)
end

function Base.any(input::NDArray{Bool}) 
    out = cuNumeric.zeros(Bool)
    return nda_unary_reduction(out, cuNumeric.ANY, input)
end

#! ONLY ADD ONCE REDUCTIONS RETURN A SCALAR
# function StatsBase.mean(arr::NDArray{T}) where T
#     return sum(arr) ./ prod(size(arr))
# end

# function Base.reduce(f::Function, arr::NDArray)
#     return f(arr)
# end


#* TODO Overload broadcasting to just call this
#* e.g. sin.(ndarray) should call this or the proper generated func
function Base.map(f::Function, arr::NDArray)
    return f.(arr) # Will try to call one of the functions generated above
end
