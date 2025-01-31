export NDArray, LegateType


#= Copyright 2025 Northwestern University, 
 *                   Carnegie Mellon University University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author(s): David Krasowska <krasow@u.northwestern.edu>
 *            Ethan Meitz <emeitz@andrew.cmu.edu>
=#

# is legion Complex128 same as ComplexF64 in julia? 
# These are methods that return a Legate::Type
global const type_map = Dict{Type, Function}(
    Bool => cuNumeric.bool_, 
    Int8 => cuNumeric.int8,
    Int16 => cuNumeric.int16,
    Int32 => cuNumeric.int32,
    Int64 => cuNumeric.int64,
    UInt8 => cuNumeric.uint8,
    UInt16 => cuNumeric.uint16,
    UInt32 => cuNumeric.uint32, 
    UInt64 => cuNumeric.uint64, 
    Float16 => cuNumeric.float16, 
    Float32 => cuNumeric.float32, 
    Float64 => cuNumeric.float64,
    # ComplexF16 => cuNumeric.complex32,  #COMMENTED OUT IN WRAPPER
    ComplexF32 => cuNumeric.complex64, 
    ComplexF64 => cuNumeric.complex128
)

# hate this but casting to Int gets around 
# a bug where TypeCode can't be used at compile time 
global const code_type_map = Dict{Int, Type}(
    Int(cuNumeric.BOOL) => Bool,
    Int(cuNumeric.INT8) => Int8,
    Int(cuNumeric.INT16) => Int16,
    Int(cuNumeric.INT32) => Int32,
    Int(cuNumeric.INT64) => Int64,
    Int(cuNumeric.UINT8) => UInt8,
    Int(cuNumeric.UINT16) => UInt16,
    Int(cuNumeric.UINT32) => UInt32,
    Int(cuNumeric.UINT64) => UInt64,
    Int(cuNumeric.FLOAT16) => Float16,
    Int(cuNumeric.FLOAT32) => Float32,
    Int(cuNumeric.FLOAT64) => Float64,
    Int(cuNumeric.COMPLEX64) => ComplexF32,
    Int(cuNumeric.COMPLEX128) => ComplexF64,
    Int(cuNumeric.STRING) => String # CxxString?
)

#probably some way to enforce this only gets passed int types
to_cpp_dims(dims::Dims{N}, int_type::Type = UInt64) where N = StdVector(int_type.([d for d in dims]))
to_cpp_dims_int(dims::Dims{N}, int_type::Type = Int64) where N = StdVector(int_type.([d for d in dims]))
to_cpp_dims_int(d::Int64, int_type::Type = Int64) = StdVector(int_type.([d]))

#julia is 1 indexed vs c is 0 indexed. added the -1 
to_cpp_index(idx::Dims{N}, int_type::Type = UInt64) where N = StdVector(int_type.([e - 1 for e in idx]))
to_cpp_index(d::Int64, int_type::Type = UInt64) = StdVector(int_type.([d - 1]))

# disgustingggggg
Base.eltype(arr::NDArray) = code_type_map[Int(code(type(arr)))]
to_legate_type(T::Type) = type_map[T]()


# probably belongs in another file
function LegateType(T::Type)
    return type_map[T]()
end


#### ARRAY/INDEXING INTERFACE ####
# https://docs.julialang.org/en/v1/manual/interfaces/#Indexing

function Base.getindex(arr::NDArray, idxs::Vararg{Int, N}) where N
    T = eltype(arr)
    acc = NDArrayAccessor{T,N}()
    return read(acc, arr, to_cpp_index(idxs))
end


function Base.setindex!(arr::NDArray, value::T, idxs::Vararg{Int, N}) where {T <: Number, N}
    acc = NDArrayAccessor{T,N}()
    write(acc, arr, to_cpp_index(idxs), value)
end

# USED TO CONVERT NDArray to Julia Array
# Long term probably be a named function since we allocate
# whole new array in here. Not exactly what I expect form []
function Base.getindex(arr::NDArray, c::Vararg{Colon, N}) where N
    arr_dims = Int.(cuNumeric.shape(arr))
    T = eltype(arr)
    julia_array = Base.zeros(T, arr_dims...)

    for CI in CartesianIndices(julia_array)
        julia_array[CI] = arr[Tuple(CI)...]
    end

    return julia_array
end

# This should also probably be a named function
# We can just define a specialization for Base.fill(::NDArray)
function Base.setindex!(arr::NDArray, val::Union{Float32, Float64}, c::Vararg{Colon, N}) where N
    fill(arr, LegateScalar(val))
end

Base.firstindex(arr::NDArray, dim::Int) = 1

function Base.lastindex(arr::NDArray, dim::Int)
    return Base.size(arr,dim)
end

Base.IndexStyle(::NDArray) = IndexCartesian()

function Base.ndims(arr::NDArray)
    return Int(cuNumeric.dim(arr))
end

function Base.size(arr::NDArray)
    return Tuple(Int.(cuNumeric.shape(arr)))
end

function Base.size(arr::NDArray, dim::Int)
    return Base.size(arr)[dim]
end

function Base.show(io::IO, arr::NDArray)
    T = eltype(arr)
    dim = Base.size(arr)
    print(io, "NDArray of $(T)s, Dim: $(dim)")
end

function Base.show(io::IO, ::MIME"text/plain", arr::NDArray)
    T = eltype(arr)
    dim = Base.size(arr)
    print(io, "NDArray of $(T)s, Dim: $(dim)")
end


#### INITIALIZATION ####

#* is this type piracy?
"""
    cuNumeric.zeros([T=Float64,] dims::Int...)
    cuNumeric.zeros([T=Float64,] dims::Tuple)

Create an NDArray with element type `T`, of all zeros with size specified by `dims`.
This function has the same signature as `Base.zeros`, so be sure to call it as `cuNuermic.zeros`.

# Examples
```jldoctest
julia> cuNumeric.zeros(2,2)
NDArray of Float64s, Dim: [2, 2]

julia> cuNumeric.zeros(Float32, 3)
NDArray of Float32s, Dim: [3]

julia> cuNumeric.zeros(Int32,(2,3))
NDArray of Int32s, Dim: [2, 3]
```
"""
function zeros(::Type{T}, dims::Dims{N}) where {N,T}
    LT = to_legate_type(T)
    opt = StdOptional{LegateType}(LT)
    dims_uint64 = to_cpp_dims(dims)
    return _zeros(dims_uint64, opt)
end

function zeros(::Type{T}, dims::Int...) where T
    return zeros(T, dims)
end

function zeros(dims::Dims{N}) where N
    return zeros(Float64, dims)
end

function zeros(dims::Int...)
    return zeros(Float64, dims)
end


function full(dims::Dims{N}, val::Union{Float32, Float64}) where N
    dims_uint64 = to_cpp_dims(dims)
    return _full(dims_uint64, LegateScalar(val))
end


"""
    rand!(arr::NDArray)

Fills `arr` with Float64s uniformly at random
"""

# This integer is unused but should represent, uniform, normal etc
Random.rand!(arr::NDArray) = cuNumeric.random(arr, 0)


#* THIS JUST COMPLETELY OVERRIDES ALL CALLS TO RAND....
"""
    rand(NDArray, dims::Dims)
    rand(NDArray, dims::Int...)

Create a new NDArray of size `dims`, filled with Float64s uniformly at random
"""
Random.rand(::Type{NDArray}, dims::Dims) = cuNumeric._random_ndarray(to_cpp_dims(dims))
Random.rand(::Type{NDArray}, dims::Int...) = cuNumeric.rand(NDArray, dims)


#### OPERATIONS ####

function reshape(arr::NDArray, i::Dims{N}) where N
    i_int64 = to_cpp_dims_int(i)
    return _reshape(arr, i_int64)
end

function reshape(arr::NDArray, i::Int64)
    i_int64 = to_cpp_dims_int(i)
    return _reshape(arr, i_int64)
end

# For matricies some might expect this to be matmul
# should probably only call this when .* is used
function Base.:*(arr1::NDArray, arr2::NDArray)
    return multiply(arr1, arr2)
end

function Base.:+(arr1::NDArray, arr2::NDArray)
    return add(arr1, arr2)
end

function Base.:+(val::Union{Float32, Float64}, arr::NDArray)
    return add_scalar(arr, LegateScalar(val))
end

function Base.:+(arr::NDArray, val::Union{Float32, Float64})
    return add_scalar(arr, LegateScalar(val))
end

function Base.:*(val::Union{Float32, Float64}, arr::NDArray)
    return multiply_scalar(arr, LegateScalar(val))
end

#* Can't overload += in Julia, this should be called by .+= 
#* to maintain some semblence native Julia array syntax
# See https://docs.julialang.org/en/v1/manual/interfaces/#extending-in-place-broadcast-2
function add!(out::NDArray, arr1::NDArray, arr2::NDArray)
    _add(arr1, arr2, out)
    return out
end

function multiply!(out::NDArray, arr1::NDArray, arr2::NDArray)
    _multiply(arr1, arr2, out)
    return out
end

function LinearAlgebra.mul!(out::NDArray, A::NDArray, B::NDArray)
    _dot_three_arg(out, A, B)
    return out
end

# arr1 == arr2
function Base.:(==)(arr1::NDArray, arr2::NDArray)
    # TODO this only works on 2D arrays
    # should we use a lazy hashing approach? 
    # something like this? would this be better than looping thru the elements?
    # hash(arr1.data) == hash(arr2.data)
    if (Base.size(arr1) != Base.size(arr2))
        return false
    end

    if(ndims(arr1) > 3)
        @warn "Accessors do not support dimension > 3 yet"
        return false
    end

    dims = Base.size(arr1)

    for CI in CartesianIndices(dims)
        if arr1[Tuple(CI)...] != arr2[Tuple(CI)...]
            return false
        end
    end

    return true
end

# arr == julia_array
function Base.:(==)(arr::NDArray, julia_array::Array)
    if (Base.size(arr) != Base.size(julia_array))
        @warn "Left NDArray is $(size(arr)) and right Julia array is $(Base.size(julia_array))!\n"
        return false
    end

    for CI in CartesianIndices(julia_array)
        if julia_array[CI] != arr[Tuple(CI)...]
            return false
        end
    end

    # successful completion
    return true
end


# julia_array == arr
function Base.:(==)(julia_array::Array, arr::NDArray)
    # flip LHS and RHS
    return (arr == julia_array)
end
