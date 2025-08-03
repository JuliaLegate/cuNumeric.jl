export NDArray

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

Base.Broadcast.broadcastable(v::NDArray) = v

const Scalar = Union{Float32,Float64,Int64,Int32}

@doc"""
    to_cpp_index(idx::Dims{N}, int_type::Type=UInt64) where {N}

**Internal API**

Converts a Julia 1-based index tuple `idx` to a zero-based C++ style index wrapped in `StdVector` of the specified integer type.

Each element of `idx` is decremented by 1 to adjust from Julia’s 1-based indexing to C++ 0-based indexing.
"""
function to_cpp_index(idx::Dims{N}, int_type::Type=UInt64) where {N}
    StdVector(int_type.([e - 1 for e in idx]))
end

@doc"""
    to_cpp_index(d::Int64, int_type::Type=UInt64)

**Internal API**

Converts a single Julia 1-based index `d` to a zero-based C++ style index wrapped in `StdVector`.
"""
to_cpp_index(d::Int64, int_type::Type=UInt64) = StdVector(int_type.([d - 1]))

@doc"""
    Base.eltype(arr::NDArray)

Returns the element type of the `NDArray`.

**Internal API**

This method uses `nda_array_type_code` internally to map to the appropriate Julia element type.
"""
Base.eltype(arr::NDArray) = Legate.code_type_map[nda_array_type_code(arr)]

@doc"""
    LegateType(T::Type)

Converts a Julia type `T` to the corresponding Legate type.

**Internal API**
"""
LegateType(T::Type) = Legate.to_legate_type(T)

@doc"""
    as_type(arr::NDArray, t::Type{T}) where {T}

Convert the element type of `arr` to type `T`, returning a new `NDArray` with elements cast to `T`.

# Arguments
- `arr::NDArray`: Input array.
- `t::Type{T}`: Target element type.

# Returns
A new `NDArray` with the same shape as `arr` but with elements of type `T`.

# Examples
```@repl
arr = cuNumeric.rand(4, 5);
as_type(arr, Float32)
```
"""
as_type(arr::NDArray, t::Type{T}) where {T} = nda_astype(arr, t)

#### ARRAY/INDEXING INTERFACE ####
# https://docs.julialang.org/en/v1/manual/interfaces/#Indexing
@doc"""
    dim(arr::NDArray)
    Base.ndims(arr::NDArray)

Return the number of dimensions of the `NDArray`.

Both functions query the underlying cuNumeric API to get
the dimensionality of the array.

# Examples
```@repl
arr = cuNumeric.rand(2, 3, 4);
dim(arr)
ndims(arr)
```
"""

dim(arr::NDArray) = Int(cuNumeric.nda_array_dim(arr))
Base.ndims(arr::NDArray) = Int(cuNumeric.nda_array_dim(arr))
@doc"""
    Base.size(arr::NDArray)
    Base.size(arr::NDArray, dim::Int)

Return the size of the given `NDArray`.

- `Base.size(arr)` returns a tuple of dimensions of the array.
- `Base.size(arr, dim)` returns the size of the array along the specified dimension `dim`.

These override Base's size methods for the `NDArray` type,
using the underlying cuNumeric API to query array shape.

# Examples
```@repl
arr = cuNumeric.rand(3, 4, 5);
size(arr)
size(arr, 2)
```
"""
Base.size(arr::NDArray) = Tuple(Int.(cuNumeric.nda_array_shape(arr)))
Base.size(arr::NDArray, dim::Int) = Base.size(arr)[dim]
@doc"""
    Base.firstindex(arr::NDArray, dim::Int)
    Base.lastindex(arr::NDArray, dim::Int)
    Base.lastindex(arr::NDArray)

Provide the first and last valid indices along a given dimension `dim` for `NDArray`.

- `firstindex` always returns 1, since Julia arrays are 1-indexed.
- `lastindex` returns the size of the array along the specified dimension.
- `lastindex(arr)` returns the size along the first dimension.

# Examples
```@repl
arr = cuNumeric.rand(4, 5);
firstindex(arr, 2)
lastindex(arr, 2)
lastindex(arr)
```
"""
Base.firstindex(arr::NDArray, dim::Int) = 1
Base.lastindex(arr::NDArray, dim::Int) = Base.size(arr, dim)
Base.lastindex(arr::NDArray) = Base.size(arr, 1)

Base.IndexStyle(::NDArray) = IndexCartesian()

@doc"""
    Base.show(io::IO, arr::NDArray)
    Base.show(io::IO, ::MIME"text/plain", arr::NDArray)

Display a summary of the `NDArray` showing its element type and dimensions.

These methods customize how `NDArray` instances appear in the REPL and in text/plain contexts.

# Example
```@repl
arr = cuNumeric.ones(NDArray, 2, 3)
```
"""
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

#### ARRAY INDEXING AND SLICES ####
@doc"""
    slice(start::Union{Nothing,Integer}, stop::Union{Nothing,Integer})

**Internal API**

Constructs a `cuNumeric.Slice` object representing a slice with optional start and stop indices.

- If `start` or `stop` is `nothing`, the slice end is considered unbounded (`Slice::OPEN`).
- Otherwise, the slice is defined as `[start, stop]` interval (inclusive).
"""

function slice(start::Union{Nothing,Integer}, stop::Union{Nothing,Integer})
    cuNumeric.Slice(
        isnothing(start) ? 0 : 1,
        isnothing(start) ? 0 : Int64(start),
        isnothing(stop) ? 0 : 1,
        isnothing(stop) ? 0 : Int64(stop),
    )
end

@doc"""
    slice_array(slices::Vararg{Tuple{Union{Int,Nothing},Union{Int,Nothing}},N}) where {N}

**Internal API**

Constructs a vector of `cuNumeric.Slice` objects from a variable number of `(start, stop)` tuples.

Each tuple corresponds to a dimension slice, using `slice` internally.
"""
function slice_array(slices::Vararg{Tuple{Union{Int,Nothing},Union{Int,Nothing}},N}) where {N}
    v = Vector{cuNumeric.Slice}(undef, N)
    for i in 1:N
        start, stop = slices[i]
        v[i] = slice(start, stop)
    end
    return v
end

@doc"""
    arr[i, j]
    arr[i]
    arr[:, j]
    arr[i, :]
    arr[i:j, :]
    arr[:, k:l]
    arr[i:j, k:l]
    arr[:, :, ...]
    arr[...] = val
    arr[i, j] = rhs
    arr[i:j, k:l] = rhs

Overloads `Base.getindex` and `Base.setindex!` to support multidimensional indexing and slicing on `cuNumeric.NDArray`s.

Slicing supports combinations of `Int`, `UnitRange`, and `Colon()` for selecting ranges of rows and columns. 
The use of all colons (`arr[:]`, `arr[:, :]`, etc.) returns a new Julia `Array` containing a copy of the data.

Assignment also supports:
- Writing NDArray slices to NDArray regions
- Broadcasting a scalar `val::Float32` or `Float64` into a slice

# Examples
```@repl
A = cuNumeric.full((3, 3), 1.0);
A[1, 2]
A[1:2, 2:3] = cuNumeric.ones(2, 2);
A[:, 1] = 5.0;
Array(A)
```
 """
##### REGULAR ARRAY INDEXING ####
function Base.getindex(arr::NDArray, idxs::Vararg{Int,N}) where {N}
    T = eltype(arr)
    acc = NDArrayAccessor{T,N}()
    return read(acc, arr.ptr, to_cpp_index(idxs))
end

function Base.setindex!(arr::NDArray, value::T, idxs::Vararg{Int,N}) where {T<:Number,N}
    acc = NDArrayAccessor{T,N}()
    write(acc, arr.ptr, to_cpp_index(idxs), value)
end

#### START OF SLICING ####
function Base.setindex!(lhs::NDArray, rhs::NDArray, i::Colon, j::Int64)
    s = nda_get_slice(lhs, slice_array((0, Base.size(lhs, 1)), (j-1, j)))
    nda_assign(s, rhs);
end

function Base.setindex!(lhs::NDArray, rhs::NDArray, i::Int64, j::Colon)
    s = nda_get_slice(lhs, slice_array((i-1, i)))
    nda_assign(s, rhs);
end

function Base.setindex!(lhs::NDArray, rhs::NDArray, i::UnitRange, j::Colon)
    s = nda_get_slice(lhs, slice_array((first(i) - 1, last(i)), (0, Base.size(lhs, 2))))
    nda_assign(s, rhs)
end

function Base.setindex!(lhs::NDArray, rhs::NDArray, i::Colon, j::UnitRange)
    s = nda_get_slice(lhs, slice_array((0, Base.size(lhs, 1)), (first(j) - 1, last(j))))
    nda_assign(s, rhs)
end

function Base.setindex!(lhs::NDArray, rhs::NDArray, i::UnitRange, j::Int64)
    s = nda_get_slice(lhs, slice_array((first(i) - 1, last(i)), (j-1, j)))
    nda_assign(s, rhs)
end

function Base.setindex!(lhs::NDArray, rhs::NDArray, i::Int64, j::UnitRange)
    s = nda_get_slice(lhs, slice_array((i-1, i), (first(j) - 1, last(j))))
    nda_assign(s, rhs)
end

function Base.setindex!(lhs::NDArray, rhs::NDArray, i::UnitRange, j::UnitRange)
    s = nda_get_slice(lhs, slice_array((first(i) - 1, last(i)), (first(j) - 1, last(j))))
    nda_assign(s, rhs)
end

function Base.getindex(arr::NDArray, i::Colon, j::Int64)
    return nda_get_slice(arr, slice_array((0, Base.size(arr, 1)), (j-1, j)))
end

function Base.getindex(arr::NDArray, i::Int64, j::Colon)
    return nda_get_slice(arr, slice_array((i-1, i)))
end

function Base.getindex(arr::NDArray, i::UnitRange, j::Colon)
    return nda_get_slice(
        arr, slice_array((first(i) - 1, last(i)), (0, Base.size(arr, 2)))
    )
end

function Base.getindex(arr::NDArray, i::Colon, j::UnitRange)
    return nda_get_slice(
        arr, slice_array((0, Base.size(arr, 1)), (first(j) - 1, last(j)))
    )
end

function Base.getindex(arr::NDArray, i::UnitRange, j::Int64)
    return nda_get_slice(arr, slice_array((first(i) - 1, last(i)), (j-1, j)))
end

function Base.getindex(arr::NDArray, i::Int64, j::UnitRange)
    return nda_get_slice(arr, slice_array((i-1, i), (first(j) - 1, last(j))))
end

function Base.getindex(arr::NDArray, i::UnitRange, j::UnitRange)
    return nda_get_slice(
        arr, slice_array((first(i) - 1, last(i)), (first(j) - 1, last(j)))
    )
end

function Base.getindex(arr::NDArray, i::UnitRange)
    return nda_get_slice(
        arr, slice_array((first(i) - 1, last(i)))
    )
end

# USED TO CONVERT NDArray to Julia Array
# Long term probably be a named function since we allocate
# whole new array in here. Not exactly what I expect form []
function Base.getindex(arr::NDArray, c::Vararg{Colon,N}) where {N}
    arr_dims = Int.(cuNumeric.nda_array_shape(arr))
    T = eltype(arr)
    julia_array = Base.zeros(T, arr_dims...)

    for CI in CartesianIndices(julia_array)
        julia_array[CI] = arr[Tuple(CI)...]
    end

    return julia_array
end

# This should also probably be a named function
# We can just define a specialization for Base.fill(::NDArray)
function Base.setindex!(arr::NDArray, val::Union{Float32,Float64}, c::Vararg{Colon,N}) where {N}
    nda_fill_array(arr, val)
end

function Base.setindex!(arr::NDArray, val::Union{Float32,Float64}, i::Colon, j::Int64)
    s = nda_get_slice(arr, to_cpp_init_slice(slice(0, Base.size(arr, 1)), slice(j-1, j)))
    nda_fill_array(s, val)
end

function Base.setindex!(arr::NDArray, val::Union{Float32,Float64}, i::Int64, j::Colon)
    s = nda_get_slice(arr, to_cpp_init_slice(slice(i-1, i)))
    nda_fill_array(s, val)
end

#### INITIALIZATION OF NDARRAYS ####
@doc"""
    cuNumeric.full(dims::Tuple, val)
    cuNumeric.full(dim::Int, val)

Create an `NDArray` filled with the scalar value `val`, with the shape specified by `dims`.

# Examples
```@repl
cuNumeric.full((2, 3), 7.5)
cuNumeric.full(4, 0)
```
"""
function full(dims::Dims{N}, val::T) where {T,N}
    shape = UInt64.(collect(dims))
    return nda_full_array(shape, val)
end

function full(dim::Int, val::T) where {T}
    shape = UInt64[dim]
    return nda_full_array(shape, val)
end

@doc"""
    cuNumeric.zeros([T=Float64,] dims::Int...)
    cuNumeric.zeros([T=Float64,] dims::Tuple)

Create an NDArray with element type `T`, of all zeros with size specified by `dims`.
This function mirrors the signature of `Base.zeros`, and defaults to `Float64` when the type is omitted.

# Examples
```@repl
cuNumeric.zeros(2, 2)
cuNumeric.zeros(Float32, 3)
cuNumeric.zeros(Int32, (2,3))
```
"""
function zeros(::Type{T}, dims::Dims{N}) where {N,T}
    shape = UInt64.(collect(dims))
    return nda_zeros_array(shape; type=T)
end

function zeros(::Type{T}, dims::Int...) where {T}
    return zeros(T, dims)
end

function zeros(dims::Dims{N}) where {N}
    return zeros(Float64, dims)
end

function zeros(dims::Int...)
    return zeros(Float64, dims)
end

@doc"""
    cuNumeric.ones([T=Float64,] dims::Int...)
    cuNumeric.ones([T=Float64,] dims::Tuple)

Create an NDArray with element type `T`, of all zeros with size specified by `dims`.
This function has the same signature as `Base.ones`, so be sure to call it as `cuNuermic.ones`.

# Examples
```@repl
cuNumeric.ones(2, 2)
cuNumeric.ones(Float32, 3)
cuNumeric.ones(Int32, (2, 3))
```
"""
function ones(::Type{T}, dims::Dims) where {T}
    return full(dims, T(1))
end

function ones(::Type{T}, dims::Int...) where {T}
    return ones(T, dims)
end

function ones(dims::Dims{N}) where {N}
    return ones(Float64, dims)
end

function ones(dims::Int...)
    return ones(Float64, dims)
end

@doc"""
    cuNumeric.rand!(arr::NDArray)

Fills `arr` with Float64s uniformly at random.

    cuNumeric.rand(NDArray, dims::Int...)
    cuNumeric.rand(NDArray, dims::Tuple)

Create a new `NDArray` of element type Float64, filled with uniform random values.

This function uses the same signature as `Base.rand` with a custom backend,
and currently supports only `Float64` with uniform distribution (`code = 0`).

# Examples
```@repl
cuNumeric.rand(NDArray, 2, 2)
cuNumeric.rand(NDArray, (4, 1))
A = cuNumeric.zeros(2, 2); cuNumeric.rand!(A)
```
"""
Random.rand!(arr::NDArray) = cuNumeric.nda_random(arr, 0)
Random.rand(::Type{NDArray}, dims::Dims) = cuNumeric.nda_random_array(UInt64.(collect(dims)))
Random.rand(::Type{NDArray}, dims::Int...) = cuNumeric.rand(NDArray, dims)

random(::Type{T}, dims::Dims) where {T} = cuNumeric.nda_random_array(UInt64.(collect(dims)))
random(::Type{T}, dim::Int64) where {T} = cuNumeric.random(T, (dim,))
random(dims::Dims, e::Type{T}) where {T} = cuNumeric.rand(e, dims)
random(arr::NDArray, code::Int64) = cuNumeric.nda_random(arr, code)
#### OPERATIONS ####
@doc"""
    reshape(arr::NDArray, dims::Dims{N}; copy::Bool = false) where {N}
    reshape(arr::NDArray, dim::Int64; copy::Bool = false)

Return a new `NDArray` reshaped to the specified dimensions.

# Examples
```@repl
arr = cuNumeric.ones(4, 3)
reshape(arr, (3, 4))
reshape(arr, 12)
```
"""

function reshape(arr::NDArray, i::Dims{N}; copy::Bool=false) where {N}
    reshaped = nda_reshape_array(arr, UInt64.(collect(i)))
    return copy ? copy(reshaped) : reshaped
end

function reshape(arr::NDArray, i::Int64; copy::Bool=false)
    reshaped = nda_reshape_array(arr, UInt64.([i]))
    return copy ? copy(reshaped) : reshaped
end

@doc"""
    Base.:+(arr::NDArray, val::Number)
    Base.:+(val::Number, arr::NDArray)
    Base.:+(lhs::NDArray, rhs::NDArray)

Add a scalar `val` to every element in the `NDArray` `arr`, or perform element-wise addition between two NDArrays,
returning a new `NDArray`.

Broadcasting is supported to enable element-wise addition between `NDArray` and scalars or between two NDArrays.

# Examples
```@repl
lhs + 3
3 + rhs
lhs + rhs
```
"""

function Base.:+(arr::NDArray, val::Scalar)
    return nda_add_scalar(arr, val)
end
function Base.:+(val::Scalar, arr::NDArray)
    return +(arr, val)
end

function Base.Broadcast.broadcasted(
    ::typeof(+), arr::NDArray, val::Scalar
)
    return +(arr, val)
end

function Base.Broadcast.broadcasted(
    ::typeof(+), val::Scalar, arr::NDArray
)
    return +(arr, val)
end

function Base.Broadcast.broadcasted(::typeof(+), lhs::NDArray, rhs::NDArray)
    return +(lhs, rhs)
end

@doc"""
    Base.:-(val::Number, arr::NDArray)
    Base.:-(arr::NDArray, val::Number)
    Base.:-(lhs::NDArray, rhs::NDArray)

Perform subtraction involving an `NDArray` and a scalar or between two NDArrays. 

- `val - arr` subtracts `val` by `arr`.
- `arr - val` subtracts scalar `val` from each element of `arr`.
- Element-wise subtraction is supported between two NDArrays.

Broadcasting is also supported for these operations.

# Examples
```@repl
lhs - 3
3 - rhs
lhs - rhs
```
"""
function Base.:-(val::Scalar, arr::NDArray)
    return nda_add_scalar(-arr, val)
end

function Base.:-(arr::NDArray, val::Scalar)
    return +(arr, (-1*val))
end

function Base.Broadcast.broadcasted(
    ::typeof(-), arr::NDArray, val::Scalar
)
    return -(arr, val)
end
function Base.Broadcast.broadcasted(
    ::typeof(-), val::Scalar, rhs::NDArray
)
    arr_type = eltype(rhs) # match the arr type
    lhs = full(Base.size(rhs), arr_type(val))
    return -(lhs, rhs)
end

function Base.Broadcast.broadcasted(::typeof(-), lhs::NDArray, rhs::NDArray)
    return -(lhs, rhs)
end

@doc"""
    Base.:*(val::Number, arr::NDArray)
    Base.:*(arr::NDArray, val::Number)
    Base.Broadcast.broadcasted(::typeof(*), arr::NDArray, val::Number)
    Base.Broadcast.broadcasted(::typeof(*), val::Number, arr::NDArray)
    Base.Broadcast.broadcasted(::typeof(*), lhs::NDArray, rhs::NDArray)

Multiply an `NDArray` by a scalar or perform element-wise multiplication between NDArrays.

- Scalar multiplication supports types: `Float32`, `Float64`, `Int32`, `Int64`.
- Broadcasting works seamlessly with scalars and NDArrays.

# Examples
```@repl
lhs * 3
2 * rhs
lhs - rhs
```
"""

function Base.:*(val::Scalar, arr::NDArray)
    return nda_multiply_scalar(arr, val)
end

function Base.:*(arr::NDArray, val::Scalar)
    return *(val, arr)
end

function Base.Broadcast.broadcasted(
    ::typeof(*), arr::NDArray, val::Scalar
)
    return *(val, arr)
end

function Base.Broadcast.broadcasted(
    ::typeof(*), val::Scalar, arr::NDArray
)
    return *(val, arr)
end

function Base.Broadcast.broadcasted(::typeof(*), lhs::NDArray, rhs::NDArray)
    return *(lhs, rhs)
end

@doc"""
    Base.:/(arr::NDArray, val::Scalar)
    Base.Broadcast.broadcasted(::typeof(/), arr::NDArray, val::Scalar)

Returns the element-wise multiplication of `arr` by the scalar reciprocal `1 / val`.

# Examples
```@repl
arr = cuNumeric.ones(2, 2)
arr / 2
```
"""
function Base.:/(arr::NDArray, val::Scalar)
    # throw(ErrorException("[/] is not supported yet"))
    return nda_multiply_scalar(arr, Float64(1 / val))
end

function Base.Broadcast.broadcasted(
    ::typeof(/), arr::NDArray, val::Scalar
)
    return nda_multiply_scalar(arr, Float64(1 / val))
end

@doc"""
    Base.Broadcast.broadcasted(::typeof(/), val::Scalar, arr::NDArray)

Throws an error since element-wise division of a scalar by an NDArray is not supported yet.

# Examples
```@repl
arr = cuNumeric.ones(2, 2)
# 2 ./ arr # ERROR
```
"""
function Base.Broadcast.broadcasted(
    ::typeof(/), val::Scalar, arr::NDArray
)
    return throw(ErrorException("element wise [val ./ NDArray] is not supported yet"))
end

@doc"""
    Base.Broadcast.broadcasted(::typeof(/), lhs::NDArray, rhs::NDArray)

Perform element-wise division of two NDArrays.

# Examples
```@repl
A = cuNumeric.rand(2, 2)
B = cuNumeric.ones(2, 2)
C = A ./ B
typeof(C)
```
"""
function Base.Broadcast.broadcasted(::typeof(/), lhs::NDArray, rhs::NDArray)
    return /(lhs, rhs)
end

#* Can't overload += in Julia, this should be called by .+= 
#* to maintain some semblence native Julia array syntax
# See https://docs.julialang.org/en/v1/manual/interfaces/#extending-in-place-broadcast-2

@doc"""
    add!(out::NDArray, arr1::NDArray, arr2::NDArray)

Compute element-wise addition of `arr1` and `arr2` storing the result in `out`.

This is an in-place operation and is used to support `.+=` style syntax.

# Examples
```@repl
a = cuNumeric.ones(2, 2)
b = cuNumeric.ones(2, 2)
out = similar(a)
add!(out, a, b)
```
"""
function add!(out::NDArray, arr1::NDArray, arr2::NDArray)
    return nda_add(arr1, arr2, out)
end

@doc"""
    multiply!(out::NDArray, arr1::NDArray, arr2::NDArray)

Compute element-wise multiplication of `arr1` and `arr2`, storing the result in `out`.

This function performs the operation in-place, modifying `out`.

# Examples
```@repl
a = cuNumeric.ones(2, 2)
b = cuNumeric.ones(2, 2)
out = similar(a)
multiply!(out, a, b)
```
"""
function multiply!(out::NDArray, arr1::NDArray, arr2::NDArray)
    return nda_multiply(arr1, arr2, out)
end

@doc"""
    LinearAlgebra.mul!(out::NDArray, arr1::NDArray, arr2::NDArray)

Compute the matrix multiplication (dot product) of `arr1` and `arr2`, storing the result in `out`.

This function performs the operation in-place, modifying `out`.

# Examples
```@repl
a = cuNumeric.ones(2, 3)
b = cuNumeric.ones(3, 2)
out = cuNumeric.zeros(2, 2)
LinearAlgebra.mul!(out, a, b)
```
"""
function LinearAlgebra.mul!(out::NDArray, arr1::NDArray, arr2::NDArray)
    return nda_three_dot_arg(arr1, arr2, out)
end

@doc"""
    Base.copy(arr::NDArray)

Create and return a deep copy of the given `NDArray`.

# Examples
```@repl
a = cuNumeric.ones(2, 2)
b = copy(a)
b === a
b[1,1] == a[1,1]
```
"""
function Base.copy(arr::NDArray)
    return nda_copy(arr)
end

@doc"""
    assign(arr::NDArray, other::NDArray)

Assign the contents of `other` to `arr` element-wise.

This function overwrites the data in `arr` with the values from `other`.  
Both arrays must have the same shape.

# Examples
```@repl
a = cuNumeric.zeros(2, 2)
b = cuNumeric.ones(2, 2)
cuNumeric.assign(a, b);
a[1,1]
```
"""
assign(arr::NDArray, other::NDArray) = nda_assign(arr, other)

@doc"""
    ==(arr1::NDArray, arr2::NDArray)

Check if two NDArrays are equal element-wise.

Returns `true` if both arrays have the same shape and all corresponding elements are equal.
Currently supports arrays up to 3 dimensions. For higher dimensions, returns `false` with a warning.

# Examples
```@repl
a = cuNumeric.ones(2, 2)
b = cuNumeric.ones(2, 2)
a == b
c = cuNumeric.zeros(2, 2)
a == c
```
"""
function Base.:(==)(arr1::NDArray, arr2::NDArray)
    if (Base.size(arr1) != Base.size(arr2))
        @warn "lhs has size $(Base.size(arr)) and rhs has size $(Base.size(arr2))!\n"
        return false
    end

    if (ndims(arr1) > 3)
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

@doc"""
    ==(arr::NDArray, julia_arr::Array)
    ==(julia_arr::Array, arr::NDArray)

Compare an `NDArray` and a Julia `Array` for element-wise equality.

Returns `true` if both arrays have the same shape and all corresponding elements are equal.
Returns `false` otherwise (including if sizes differ, with a warning).

The second method simply calls the first with flipped arguments.

# Examples
```@repl
arr = cuNumeric.ones(2, 2)
julia_arr = ones(2, 2)
arr == julia_arr
julia_arr == arr
julia_arr2 = zeros(2, 2)
arr == julia_arr2
```
"""
function Base.:(==)(arr::NDArray, julia_array::Array)
    if (Base.size(arr) != Base.size(julia_array))
        @warn "NDArray has size $(Base.size(arr)) and Julia array has size $(Base.size(julia_array))!\n"
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

@doc"""
    compare(x, y, max_diff)

**Internal API**

Compare two arrays `x` and `y` for approximate equality within a maximum difference `max_diff`.

Supports comparisons between:
- an `NDArray` and a Julia `AbstractArray`
- two `NDArray`s
- a Julia `AbstractArray` and an `NDArray`

Returns `true` if the arrays have the same shape and element type (for mixed types),
and all corresponding elements differ by no more than `max_diff`.

Emits warnings when array sizes or element types differ.

# Notes
- This is an internal API used by higher-level approximate equality functions.
- Does not support relative tolerance (`rtol`).

# Behavior
- Checks size compatibility.
- Checks element type compatibility for `NDArray` vs Julia array.
- Iterates over elements using `CartesianIndices` to compare element-wise difference.
"""
function compare(julia_array::AbstractArray, arr::NDArray, max_diff)
    if (Base.size(arr) != Base.size(julia_array))
        @warn "NDArray has size $(Base.size(arr)) and Julia array has size $(Base.size(julia_array))!\n"
        return false
    end

    if (eltype(arr) != eltype(julia_array))
        @warn "NDArray has eltype $(eltype(arr)) and Julia array has eltype $(eltype(julia_array))!\n"
        return false
    end

    for CI in CartesianIndices(julia_array)
        if abs(julia_array[CI] - arr[Tuple(CI)...]) > max_diff
            return false
        end
    end

    # successful completion
    return true
end

function compare(arr::NDArray, julia_array::AbstractArray, max_diff)
    return compare(julia_array, arr, max_diff)
end

function compare(arr::NDArray, arr2::NDArray, max_diff)
    if (Base.size(arr) != Base.size(arr2))
        @warn "NDArray LHS has size $(Base.size(arr)) and NDArray RHS has size $(Base.size(arr2))!\n"
        return false
    end

    dims = Base.size(arr)
    for CI in CartesianIndices(dims)
        if abs(arr2[Tuple(CI)...] - arr[Tuple(CI)...]) > max_diff
            return false
        end
    end

    # successful completion
    return true
end

@doc"""
    isapprox(arr1::NDArray, arr2::NDArray; atol=0, rtol=0)
    isapprox(arr::NDArray, julia_array::AbstractArray; atol=0, rtol=0)
    isapprox(julia_array::AbstractArray, arr::NDArray; atol=0, rtol=0)

Approximate equality comparison between two `NDArray`s or between an `NDArray` and a Julia `AbstractArray`.

Returns `true` if the arrays have the same shape and all corresponding elements are approximately equal
within the given absolute tolerance `atol` and relative tolerance `rtol`.

The second and third methods handle comparisons between `NDArray` and Julia arrays by forwarding to
a common comparison function.

# Examples
```@repl
arr1 = cuNumeric.ones(2, 2)
arr2 = cuNumeric.ones(2, 2)
julia_arr = ones(2, 2)
isapprox(arr1, arr2)
isapprox(arr1, julia_arr)
isapprox(julia_arr, arr2)
```
"""
function Base.isapprox(julia_array::AbstractArray, arr::NDArray; atol=0, rtol=0)
    return compare(julia_array, arr, atol)
end

function Base.isapprox(arr::NDArray, julia_array::AbstractArray; atol=0, rtol=0)
    return compare(julia_array, arr, atol)
end

function Base.isapprox(arr::NDArray, arr2::NDArray; atol=0, rtol=0)
    return compare(arr, arr2, atol)
end
