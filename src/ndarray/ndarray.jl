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

export unwrap

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
Base.copy(arr::NDArray) = nda_copy(arr)

@doc"""
    copyto!(arr::NDArray, other::NDArray)

Assign the contents of `other` to `arr` element-wise.

This function overwrites the data in `arr` with the values from `other`.  
Both arrays must have the same shape.

# Examples
```@repl
a = cuNumeric.zeros(2, 2)
b = cuNumeric.ones(2, 2)
copyto!(a, b);
a[1,1]
```
"""
Base.copyto!(arr::NDArray{T,N}, other::NDArray{T,N}) where {T,N} = nda_assign(arr, other)

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
as_type(arr::NDArray{S,N}, ::Type{T}) where {S,T,N} = nda_astype(arr, T)::NDArray{T,N}
as_type(arr::NDArray{T}, ::Type{T}) where {T} = arr

# conversion from NDArray to Base Julia array
function (::Type{<:Array{A}})(arr::NDArray{B}) where {A,B}
    assertscalar("Array(...)") #! CAN WE DO THIS WITHOUT SCALAR INDEXING??
    dims = Base.size(arr)
    out = Base.zeros(A, dims)
    for CI in CartesianIndices(dims)
        out[CI] = A(arr[Tuple(CI)...])
    end
    return out
end

function (::Type{<:Array})(arr::NDArray{B}) where {B}
    assertscalar("Array(...)") #! CAN WE DO THIS WITHOUT SCALAR INDEXING??
    dims = Base.size(arr)
    out = Base.zeros(B, dims)
    for CI in CartesianIndices(dims)
        out[CI] = arr[Tuple(CI)...]
    end
    return out
end

# conversion from Base Julia array to NDArray
function (::Type{<:NDArray{A}})(arr::Array{B}) where {A,B}
    assertscalar("Array(...)") #! CAN WE DO THIS WITHOUT SCALAR INDEXING??
    dims = Base.size(arr)
    out = cuNumeric.zeros(A, dims)
    for CI in CartesianIndices(dims)
        out[Tuple(CI)...] = A(arr[CI])
    end
    return out
end

function (::Type{<:NDArray})(arr::Array{B}) where {B}
    assertscalar("Array(...)") #! CAN WE DO THIS WITHOUT SCALAR INDEXING??
    dims = Base.size(arr)
    out = cuNumeric.zeros(B, dims)
    for CI in CartesianIndices(dims)
        out[Tuple(CI)...] = arr[CI]
    end
    return out
end

# Base.convert(::Type{<:NDArray{T}}, a::A) where {T, A} = NDArray(T(a))::NDArray{T}
# Base.convert(::Type{T}, a::T) where {T <: NDArray} = a

# #! NEED TO THROW ERROR ON PROMOTION TO DOUBLE PRECISION??
# #! ADD MECHANISM LIKE @allowscalar, @allowdouble ??
# Base.convert(::Type{NDArray{T}}, a::NDArray) where {T} = as_type(copy(a), T)
# Base.convert(::Type{NDArray{T,N}}, a::NDArray{<:Any,N}) where {T,N} = as_type(copy(a), T)

#### ARRAY/INDEXING INTERFACE ####
# https://docs.julialang.org/en/v1/manual/interfaces/#Indexing

@doc"""
    Base.eltype(arr::NDArray)

Returns the element type of the `NDArray`.
"""
Base.eltype(arr::NDArray{T}) where {T} = T

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

dim(::NDArray{T,N}) where {T,N} = N::Int
Base.ndims(::NDArray{T,N}) where {T,N} = N::Int
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
Base.size(arr::NDArray) = cuNumeric.shape(arr)
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

Base.axes(arr::NDArray) = Base.OneTo.(size(arr))
Base.view(arr::NDArray, inds...) = arr[inds...] # NDArray slices are views by default.

Base.IndexStyle(::NDArray) = IndexCartesian()

function Base.show(io::IO, arr::NDArray{T,0}) where {T}
    println(io, "0-dimensional NDArray{$(T),0}")
    print(io, arr[]) #! should I assert scalar??
end

function Base.show(io::IO, ::MIME"text/plain", arr::NDArray{T,0}) where {T}
    println(io, "0-dimensional NDArray{$(T),0}")
    print(io, arr[]) #! should I assert scalar??
end

function Base.show(io::IO, arr::NDArray{T,D}; elems=false) where {T,D}
    print(io, "NDArray of $(T)s, Dim: $(D)")
    if elems # print all elems of array
        dims = shape(arr)
        print(io, "[")
        indxs = CartesianIndices(dims)
        lastidx = last(indxs)
        for CI in indxs
            print(io, "$(arr[Tuple(CI)...])")
            if CI != lastidx
                print(io, ", ")
            end
        end
        print(io, "]")
    end
end

function Base.show(io::IO, ::MIME"text/plain", arr::NDArray{T}; elems=false) where {T}
    Base.show(io, arr; elems=elems)
end

function Base.print(arr::NDArray{T}; elems=false) where {T}
    Base.show(stdout, arr; elems=elems)
end

function Base.println(arr::NDArray{T}; elems=false) where {T}
    Base.show(stdout, arr; elems=elems)
    print("\n")
end
#### ARRAY INDEXING AND SLICES ####

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
function Base.getindex(arr::NDArray{T,N}, idxs::Vararg{Int,N}) where {T<:SUPPORTED_NUMERIC_TYPES,N}
    assertscalar("getindex")
    acc = NDArrayAccessor{T,N}()
    return read(acc, arr.ptr, to_cpp_index(idxs))
end

function Base.getindex(arr::NDArray{T,0}) where {T<:SUPPORTED_NUMERIC_TYPES}
    assertscalar("getindex")
    acc = NDArrayAccessor{T,1}()
    zero_index = StdVector([UInt64(0)]) #! CAN I PREALLOCATE THIS SOMEHOW
    return read(acc, arr.ptr, zero_index)
end

function Base.getindex(arr::NDArray{Bool,N}, idxs::Vararg{Int,N}) where {N}
    assertscalar("getindex")
    acc = NDArrayAccessor{CxxWrap.CxxBool,N}()
    return read(acc, arr.ptr, to_cpp_index(idxs))
end

function Base.getindex(arr::NDArray{Bool,0})
    assertscalar("getindex")
    acc = NDArrayAccessor{CxxWrap.CxxBool,1}()
    zero_index = StdVector([UInt64(0)]) #! CAN I PREALLOCATE THIS SOMEHOW
    return read(acc, arr.ptr, zero_index)
end

#! TODO SUPPORT CONVERSION OF VALUES
function Base.setindex!(arr::NDArray{T,N}, value::T, idxs::Vararg{Int,N}) where {T,N}
    assertscalar("setindex!")
    _setindex!(Val{N}(), arr, value, idxs...)
end

function _setindex!(::Val{0}, arr::NDArray{T,0}, value::T) where {T<:SUPPORTED_NUMERIC_TYPES}
    acc = NDArrayAccessor{T,1}()
    write(acc, arr.ptr, StdVector(UInt64[0]), value)
end

function _setindex!(::Val{0}, arr::NDArray{Bool,0}, value::Bool)
    acc = NDArrayAccessor{CxxWrap.CxxBool,1}()
    write(acc, arr.ptr, StdVector(UInt64[0]), value)
end

function _setindex!(
    ::Val{N}, arr::NDArray{T,N}, value::T, idxs::Vararg{Int,N}
) where {T<:SUPPORTED_NUMERIC_TYPES,N}
    acc = NDArrayAccessor{T,N}()
    write(acc, arr.ptr, to_cpp_index(idxs), value)
end

function _setindex!(::Val{N}, arr::NDArray{Bool,N}, value::Bool, idxs::Vararg{Int,N}) where {N}
    acc = NDArrayAccessor{CxxWrap.CxxBool,N}()
    write(acc, arr.ptr, to_cpp_index(idxs), value)
end

#### START OF SLICING ####
function Base.setindex!(lhs::NDArray, rhs::NDArray, i::Colon, j::Int64)
    s = nda_get_slice(lhs, slice_array((0, Base.size(lhs, 1)), (j-1, j)))
    copyto!(s, rhs);
end

function Base.setindex!(lhs::NDArray, rhs::NDArray, i::Int64, j::Colon)
    s = nda_get_slice(lhs, slice_array((i-1, i)))
    copyto!(s, rhs);
end

function Base.setindex!(lhs::NDArray, rhs::NDArray, i::UnitRange, j::Colon)
    s = nda_get_slice(lhs, slice_array((first(i) - 1, last(i)), (0, Base.size(lhs, 2))))
    copyto!(s, rhs)
end

function Base.setindex!(lhs::NDArray, rhs::NDArray, i::Colon, j::UnitRange)
    s = nda_get_slice(lhs, slice_array((0, Base.size(lhs, 1)), (first(j) - 1, last(j))))
    copyto!(s, rhs)
end

function Base.setindex!(lhs::NDArray, rhs::NDArray, i::UnitRange, j::Int64)
    s = nda_get_slice(lhs, slice_array((first(i) - 1, last(i)), (j-1, j)))
    copyto!(s, rhs)
end

function Base.setindex!(lhs::NDArray, rhs::NDArray, i::Int64, j::UnitRange)
    s = nda_get_slice(lhs, slice_array((i-1, i), (first(j) - 1, last(j))))
    copyto!(s, rhs)
end

function Base.setindex!(lhs::NDArray, rhs::NDArray, i::UnitRange, j::UnitRange)
    s = nda_get_slice(lhs, slice_array((first(i) - 1, last(i)), (first(j) - 1, last(j))))
    copyto!(s, rhs)
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

Base.getindex(arr::NDArray{T}, c::Vararg{Colon,N}) where {T,N} = Base.copy(arr)
function Base.setindex!(arr::NDArray{T}, rhs::NDArray{T}, c::Vararg{Colon,N}) where {T,N}
    Base.copyto!(arr, rhs)
end

function Base.setindex!(arr::NDArray{T,2}, val::T, i::Colon, j::Int64) where {T}
    s = nda_get_slice(arr, to_cpp_init_slice(slice(0, Base.size(arr, 1)), slice(j-1, j)))
    nda_fill_array(s, val)
end

function Base.setindex!(arr::NDArray{T,2}, val::T, i::Int64, j::Colon) where {T}
    s = nda_get_slice(arr, to_cpp_init_slice(slice(i-1, i)))
    nda_fill_array(s, val)
end

Base.fill!(arr::NDArray{T}, val::T) where {T} = nda_fill_array(arr, val)

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
function full(dims::Dims, val::T) where {T<:SUPPORTED_TYPES}
    shape = collect(UInt64, dims)
    return nda_full_array(shape, val)
end

function full(dim::Int, val::T) where {T<:SUPPORTED_TYPES}
    shape = UInt64[dim]
    return nda_full_array(shape, val)
end

@doc"""
    cuNumeric.trues(dims::Tuple, val)
    cuNumeric.trues(dim::Int, val)
    cuNumeric.trues(dims::Int...)

Create an `NDArray` filled with the true, with the shape specified by `dims`.

# Examples
```@repl
cuNumeric.trues(2, 3)
```
"""
trues(dim::Int) = cuNumeric.full(dim, true)
trues(dims::Dims) = cuNumeric.full(dims, true)
trues(dims::Int...) = cuNumeric.full(dims, true)

@doc"""
    cuNumeric.falses(dims::Tuple, val)
    cuNumeric.falses(dim::Int, val)
    cuNumeric.falses(dims::Int...)

Create an `NDArray` filled with the false, with the shape specified by `dims`.

# Examples
```@repl
cuNumeric.falses(2, 3)
```
"""
falses(dim::Int) = cuNumeric.full(dim, false)
falses(dims::Dims) = cuNumeric.full(dims, false)
falses(dims::Int...) = cuNumeric.full(dims, false)

@doc"""
    cuNumeric.zeros([T=Float32,] dims::Int...)
    cuNumeric.zeros([T=Float32,] dims::Tuple)

Create an NDArray with element type `T`, of all zeros with size specified by `dims`.
This function mirrors the signature of `Base.zeros`, and defaults to `Float32` when the type is omitted.

# Examples
```@repl
cuNumeric.zeros(2, 2)
cuNumeric.zeros(Float64, 3)
cuNumeric.zeros(Int32, (2,3))
```
"""
function zeros(::Type{T}, dims::Dims) where {T<:SUPPORTED_TYPES}
    shape = collect(UInt64, dims)
    return nda_zeros_array(shape, T)
end

function zeros(::Type{T}, dims::Int...) where {T<:SUPPORTED_TYPES}
    return zeros(T, dims)
end

function zeros(dims::Dims)
    return zeros(DEFAULT_FLOAT, dims)
end

function zeros(dims::Int...)
    return zeros(DEFAULT_FLOAT, dims)
end

function zeros(::Type{T}) where {T}
    return nda_zeros_array(UInt64[], T)
end

function zeros()
    return zeros(DEFAULT_FLOAT)
end

function zeros_like(arr::NDArray)
    return zeros(eltype(arr), Base.size(arr))
end

@doc"""
    cuNumeric.ones([T=Float32,] dims::Int...)
    cuNumeric.ones([T=Float32,] dims::Tuple)

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
    return ones(DEFAULT_FLOAT, dims)
end

function ones(dims::Int...)
    return ones(DEFAULT_FLOAT, dims)
end

function ones(::Type{T}) where {T}
    return full((), T(1))
end

function ones()
    return zeros(DEFAULT_FLOAT)
end

@doc"""
    cuNumeric.rand!(arr::NDArray)

Fills `arr` with AbstractFloats uniformly at random.

    cuNumeric.rand(NDArray, dims::Int...)
    cuNumeric.rand(NDArray, dims::Tuple)

Create a new `NDArray` of element type Float64, filled with uniform random values.

This function uses the same signature as `Base.rand` with a custom backend,
and currently supports only `Float64` with uniform distribution (`code = 0`).
In order to support other Floats, we type convert for the user automatically.

# Examples
```@repl
cuNumeric.rand(NDArray, 2, 2)
cuNumeric.rand(NDArray, (4, 1))
A = cuNumeric.zeros(2, 2); cuNumeric.rand!(A)
```
"""
Random.rand!(arr::NDArray{Float64}) = cuNumeric.nda_random(arr, 0)
rand(::Type{NDArray}, dims::Dims) = cuNumeric.nda_random_array(UInt64.(collect(dims)))
rand(::Type{NDArray}, dims::Int...) = cuNumeric.rand(NDArray, dims)
rand(dims::Dims) = cuNumeric.rand(NDArray, dims)
rand(dims::Int...) = cuNumeric.rand(NDArray, dims)

function rand(::Type{T}, dims::Dims) where {T<:AbstractFloat}
    arrfp64 = cuNumeric.nda_random_array(UInt64.(collect(dims)))
    # if T == Float64, as_type should do minimial work # TODO check this.
    return cuNumeric.as_type(arrfp64, T)
end

rand(::Type{T}, dims::Int...) where {T<:AbstractFloat} = cuNumeric.rand(T, dims)

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

# Ignore the scalar indexing here...
unwrap(x::NDArray{<:Any,0}) = @allowscalar x[]
unwrap(x::NDArray{<:Any,1}) = @allowscalar x[][1] # assumes 1 element

@doc"""
    ==(arr1::NDArray, arr2::NDArray)

Check if two NDArrays are equal element-wise.

Returns `true` if both arrays have the same shape and all corresponding elements are equal.
Currently supports arrays up to 3 dimensions. For higher dimensions, returns `false` with a warning.

!!! warning

    This function uses scalar indexing and should not be used in production code. This is meant for testing.



# Examples
```@repl
a = cuNumeric.ones(2, 2)
b = cuNumeric.ones(2, 2)
a == b
c = cuNumeric.zeros(2, 2)
a == c
```
"""
function Base.:(==)(arr1::NDArray{T,N}, arr2::NDArray{T,N}) where {T,N}
    return nda_array_equal(arr1, arr2) #DOESNT RETURN SCALAR
end

function Base.:(!=)(arr1::NDArray{T,N}, arr2::NDArray{T,N}) where {T,N}
    return !(arr1 == arr2)
end

@doc"""
    ==(arr::NDArray, julia_arr::Array)
    ==(julia_arr::Array, arr::NDArray)

Compare an `NDArray` and a Julia `Array` for element-wise equality.

Returns `true` if both arrays have the same shape and all corresponding elements are equal.
Returns `false` otherwise (including if sizes differ, with a warning).

!!! warning

    This function uses scalar indexing and should not be used in production code. This is meant for testing.



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
function Base.:(==)(arr::NDArray, julia_arr::Array)
    assertscalar("==")
    return julia_arr == Array(arr)
end

Base.:(==)(julia_arr::Array, arr::NDArray) = (arr == julia_arr)

@doc"""
    isapprox(arr1::NDArray, arr2::NDArray; atol=0, rtol=0)
    isapprox(arr::NDArray, julia_array::AbstractArray; atol=0, rtol=0)
    isapprox(julia_array::AbstractArray, arr::NDArray; atol=0, rtol=0)

Approximate equality comparison between two `NDArray`s or between an `NDArray` and a Julia `AbstractArray`.

Returns `true` if the arrays have the same shape and all corresponding elements are approximately equal
within the given absolute tolerance `atol` and relative tolerance `rtol`.

The second and third methods handle comparisons between `NDArray` and Julia arrays by forwarding to
a common comparison function.

!!! warning

    This function uses scalar indexing and should not be used in production code. This is meant for testing.



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
function Base.isapprox(julia_array::AbstractArray{T}, arr::NDArray{T}; atol=0, rtol=0) where {T}
    #! REPLCE THIS WITH BIN_OP isapprox
    return compare(julia_array, arr, atol, rtol)
end

function Base.isapprox(arr::NDArray{T}, julia_array::AbstractArray{T}; atol=0, rtol=0) where {T}
    return compare(julia_array, arr, atol, rtol)
end

function Base.isapprox(arr::NDArray{T}, arr2::NDArray{T}; atol=0, rtol=0) where {T}
    return compare(arr, arr2, atol, rtol)
end
