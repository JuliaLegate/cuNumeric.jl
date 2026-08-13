#= Copyright 2026 Northwestern University,
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
 *            Nader Rahhal <naderrahhal2026@u.northwestern.edu>
=#

export unwrap

# See TODO.md (Base / LinearAlgebra sections) for AbstractArray and LA gaps.

@doc"""
    cuNumeric.transpose(arr::NDArray)

Return a new `NDArray` that is the transpose of the input `arr`.
"""
function transpose(arr::NDArray)
    return nda_transpose(arr)
end

@doc"""
    cuNumeric.ravel(arr::NDArray)

Return a flattened 1D view of the input `NDArray`.
"""
function ravel(arr::NDArray)
    return nda_ravel(arr)
end

@doc"""
    cuNumeric.unique(arr::NDArray)

Return a new `NDArray` containing the unique elements of the input `arr`.
"""
function unique(arr::NDArray)
    return nda_unique(arr)
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
@inline function Base.copyto!(arr::NDArray{T,N}, other::NDArray{T,N}) where {T,N}
    nda_assign(arr, other)
    return arr
end

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

# Wrap a raw pointer into an AbstractArray view
function make_array(::Type{T}, ptr::Ptr{T}, shape::NTuple{N,Int}) where {T,N}
    return unsafe_wrap(Array{T,N}, ptr, shape; own=false)
end

# conversion from NDArray to Base Julia array
# get_ptr is a blocking call that grabs the physical store
# we have not tested across multiple processes or devices yet

# NDArray-specific overrides of Core's AbstractArray constructors (NDArray <:
# AbstractArray): exact `Array{T}` / `Array{T,N}` / `Array` signatures so we win
# over `Array{T,N}(::AbstractArray)` (which would scalar-index). Bulk path uses
# `_copy_to_julia_array`; 1-d dispatches same-type (zero-copy) vs convert.
function (::Type{Array{T}})(arr::NDArray{S,0}) where {T,S}
    out = Array{T,0}(undef)
    allowscalar() do
        return out[] = convert(T, arr[])
    end
    return out
end

function (::Type{Array{T}})(arr::NDArray{T,1}) where {T}
    return make_array(T, Ptr{T}(get_ptr(arr)), size(arr))
end

function (::Type{Array{T}})(arr::NDArray{S,1}) where {T,S}
    return T.(make_array(S, Ptr{S}(get_ptr(arr)), size(arr)))
end

# Copy logically into Julia's column-major storage.
# Legate may map an NDArray in C or Fortran order.
function _copy_to_julia_array(arr::NDArray{T,N}) where {T,N}
    out = Array{T}(undef, size(arr))
    store = Legate.attach_external_col_major(out)
    ptr = cuNumeric.nda_store_to_ndarray(store.handle)
    finalize(store.handle)
    attached = NDArray(ptr, T, Val(N), out)
    copyto!(attached, arr)
    get_ptr(attached) # Block until the copy into `out` completes.
    return out
end

function (::Type{Array{T}})(arr::NDArray{S,N}) where {T,S,N}
    out = _copy_to_julia_array(arr)
    return T === S ? out : copyto!(Array{T}(undef, size(arr)), out)
end

(::Type{Array{T,N}})(arr::NDArray{S,N}) where {T,S,N} = Array{T}(arr)

(::Type{Array})(arr::NDArray{T,N}) where {T,N} = Array{T}(arr)

# conversion from Base Julia array to NDArray
# Julia Arrays are column-major; Legate stores are row-major. For N>=2 we
# materialize a C-ordered buffer via permutedims, attach it with the original
# shape, and keep that buffer as `parent` for lifetime.
function _nda_from_julia_array(arr::Array{T,0}) where {T}
    return cuNumeric.nda_attach_external(arr)
end

function _nda_from_julia_array(arr::Array{T,1}) where {T}
    return cuNumeric.nda_attach_external(arr)
end

function _nda_from_julia_array(arr::Array{T,N}) where {T,N}
    tmp = collect(permutedims(arr, reverse(ntuple(identity, Val(N)))))
    return cuNumeric.nda_attach_external(tmp; shape=size(arr))
end

function (::Type{<:NDArray{T}})(arr::Array{T,N}) where {T,N}
    return _nda_from_julia_array(arr)
end

function (::Type{<:NDArray{A}})(arr::Array{B,N}) where {A,B,N}
    # If types differ, cast in Julia first (creating a temp) then attach
    return _nda_from_julia_array(convert(Array{A}, arr))
end

function (::Type{<:NDArray})(arr::Array{T,N}) where {T,N}
    return _nda_from_julia_array(arr)
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
    Base.size(arr::NDArray, dim::Integer)

Return the size of the given `NDArray`.

- `Base.size(arr)` returns a tuple of dimensions of the array.
- `Base.size(arr, dim)` returns the size of the array along the specified dimension `dim`.

# Examples
```@repl
arr = cuNumeric.rand(3, 4, 5);
size(arr)
size(arr, 2)
```
"""
Base.size(arr::NDArray{<:Any,N}) where {N} = cuNumeric.shape(arr)
Base.size(arr::NDArray, dim::Integer) = dim <= ndims(arr) ? size(arr)[dim] : 1
Base.isempty(arr::NDArray) = any(==(0), size(arr))
Base.length(arr::NDArray) = prod(size(arr))

@doc"""
    Base.firstindex(arr::NDArray, dim::Integer)
    Base.lastindex(arr::NDArray, dim::Integer)
    Base.lastindex(arr::NDArray)

Provide the first and last valid indices along a given dimension `dim` for `NDArray`.

# Examples
```@repl
arr = cuNumeric.rand(4, 5);
firstindex(arr, 2)
lastindex(arr, 2)
lastindex(arr)
```
"""
Base.firstindex(arr::NDArray, dim::Integer) = 1
Base.lastindex(arr::NDArray, dim::Integer) = size(arr, dim)
Base.lastindex(arr::NDArray) = length(arr)
Base.IndexStyle(::Type{<:NDArray}) = IndexCartesian()

Base.axes(arr::NDArray) = Base.OneTo.(size(arr))
Base.view(arr::NDArray, inds...) = arr[inds...] # NDArray slices are views by default.

function Base.show(io::IO, arr::NDArray{T,0}) where {T}
    print(io, summary(arr), "(")
    @allowscalar show(io, arr[])
    return print(io, ")")
end

# Used by print(arr), println(arr), and nested displays
function Base.show(io::IO, arr::NDArray)
    return show(io, Array(arr))
end

# Used for full REPL display
function Base.show(io::IO, ::MIME"text/plain", arr::NDArray)
    summary(io, arr)

    isempty(arr) && return nothing

    println(io, ":")
    return Base.print_array(io, Array(arr))
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

Slicing supports combinations of `Integer`, `UnitRange`, and `Colon()` for selecting ranges of rows and columns.
The use of all colons (`arr[:]`, `arr[:, :]`, etc.) returns a new Julia `Array` containing a copy of the data.

Assignment also supports:
- Writing NDArray slices to NDArray regions
- Broadcasting a scalar `val::Float32` or `Float64` into a slice

# Examples
```@repl
A = cuNumeric.fill(1.0, (3, 3));
A[1, 2]
A[1:2, 2:3] = cuNumeric.ones(2, 2);
A[:, 1] = 5.0;
Array(A)
```
 """
##### REGULAR ARRAY INDEXING ####
@inline function Base.getindex(
    arr::NDArray{T,N}, idxs::Vararg{Integer,N}
) where {T<:SUPPORTED_NUMERIC_TYPES,N}
    @boundscheck checkbounds(arr, idxs...)
    assertscalar("getindex")
    acc = NDArrayAccessor{T,N}()
    return read(acc, arr.ptr, to_cpp_index(Int.(idxs)))
end

function Base.getindex(arr::NDArray{T,0}) where {T<:SUPPORTED_NUMERIC_TYPES}
    assertscalar("getindex")
    acc = NDArrayAccessor{T,1}()
    zero_index = StdVector([UInt64(0)]) #! CAN I PREALLOCATE THIS SOMEHOW
    return read(acc, arr.ptr, zero_index)
end

@inline function Base.getindex(arr::NDArray{Bool,N}, idxs::Vararg{Integer,N}) where {N}
    @boundscheck checkbounds(arr, idxs...)
    assertscalar("getindex")
    acc = NDArrayAccessor{CxxWrap.CxxBool,N}()
    return read(acc, arr.ptr, to_cpp_index(Int.(idxs)))
end

function Base.getindex(arr::NDArray{Bool,0})
    assertscalar("getindex")
    acc = NDArrayAccessor{CxxWrap.CxxBool,1}()
    zero_index = StdVector([UInt64(0)]) #! CAN I PREALLOCATE THIS SOMEHOW
    return read(acc, arr.ptr, zero_index)
end

#! TODO SUPPORT CONVERSION OF VALUES
@inline function Base.setindex!(
    arr::NDArray{T,N}, value::T, idxs::Vararg{Integer,N}
) where {T,N}
    @boundscheck checkbounds(arr, idxs...)
    assertscalar("setindex!")
    return _setindex!(Val{N}(), arr, value, idxs...)
end

@inline function Base.setindex!(
    arr::NDArray{Complex{T},N}, value::T, idxs::Vararg{Integer,N}
) where {T,N}
    @boundscheck checkbounds(arr, idxs...)
    assertscalar("setindex!")
    return _setindex!(Val{N}(), arr, Complex{T}(value), idxs...)
end

@inline function Base.setindex!(
    arr::NDArray{T,N}, value, idxs::Vararg{Integer,N}
) where {T,N}
    @boundscheck checkbounds(arr, idxs...)
    assertscalar("setindex!")
    return _setindex!(Val{N}(), arr, convert(T, value), idxs...)
end

function _setindex!(::Val{0}, arr::NDArray{T,0}, value::T) where {T<:SUPPORTED_NUMERIC_TYPES}
    acc = NDArrayAccessor{T,1}()
    return write(acc, arr.ptr, StdVector(UInt64[0]), value)
end

function _setindex!(::Val{0}, arr::NDArray{Bool,0}, value::Bool)
    acc = NDArrayAccessor{CxxWrap.CxxBool,1}()
    return write(acc, arr.ptr, StdVector(UInt64[0]), value)
end

function _setindex!(
    ::Val{N}, arr::NDArray{T,N}, value::T, idxs::Vararg{Integer,N}
) where {T<:SUPPORTED_NUMERIC_TYPES,N}
    acc = NDArrayAccessor{T,N}()
    return write(acc, arr.ptr, to_cpp_index(Int.(idxs)), value)
end

function _setindex!(
    ::Val{N}, arr::NDArray{Bool,N}, value::Bool, idxs::Vararg{Integer,N}
) where {N}
    acc = NDArrayAccessor{CxxWrap.CxxBool,N}()
    return write(acc, arr.ptr, to_cpp_index(Int.(idxs)), value)
end

#### START OF SLICING ####
# LHS slices from `nda_get_slice` are invisible to `@accelerate`; destroy
# the view handle after submitting the assign so they cannot pile up under Julia
# GC (which sees each NDArray as ~pointer-sized).
function _setindex_slice!(lhs::NDArray, rhs::NDArray, slices)
    s = nda_get_slice(lhs, slices)
    copyto!(s, rhs)
    destroy!(s)
    return nothing
end

@inline _zero_based_index(i::Integer) = (Int(i) - 1, Int(i))
@inline _zero_based_range(i::AbstractUnitRange{<:Integer}) = (Int(first(i)) - 1, Int(last(i)))

@inline function Base.setindex!(
    lhs::NDArray{T,2}, rhs::NDArray, ::Colon, j::Integer
) where {T}
    @boundscheck checkbounds(lhs, :, j)
    return _setindex_slice!(
        lhs, rhs, slice_array((0, size(lhs, 1)), _zero_based_index(j))
    )
end

@inline function Base.setindex!(
    lhs::NDArray{T,2}, rhs::NDArray, i::Integer, ::Colon
) where {T}
    @boundscheck checkbounds(lhs, i, :)
    return _setindex_slice!(lhs, rhs, slice_array(_zero_based_index(i)))
end

@inline function Base.setindex!(
    lhs::NDArray{T,2}, rhs::NDArray, i::AbstractUnitRange{<:Integer}, ::Colon
) where {T}
    @boundscheck checkbounds(lhs, i, :)
    return _setindex_slice!(
        lhs, rhs, slice_array(_zero_based_range(i), (0, size(lhs, 2)))
    )
end

@inline function Base.setindex!(
    lhs::NDArray{T,2}, rhs::NDArray, ::Colon, j::AbstractUnitRange{<:Integer}
) where {T}
    @boundscheck checkbounds(lhs, :, j)
    return _setindex_slice!(
        lhs, rhs, slice_array((0, size(lhs, 1)), _zero_based_range(j))
    )
end

@inline function Base.setindex!(
    lhs::NDArray{T,2},
    rhs::NDArray,
    i::AbstractUnitRange{<:Integer},
    j::Integer,
) where {T}
    @boundscheck checkbounds(lhs, i, j)
    return _setindex_slice!(
        lhs, rhs, slice_array(_zero_based_range(i), _zero_based_index(j))
    )
end

@inline function Base.setindex!(
    lhs::NDArray{T,2},
    rhs::NDArray,
    i::Integer,
    j::AbstractUnitRange{<:Integer},
) where {T}
    @boundscheck checkbounds(lhs, i, j)
    return _setindex_slice!(
        lhs, rhs, slice_array(_zero_based_index(i), _zero_based_range(j))
    )
end

@inline function Base.setindex!(
    lhs::NDArray{T,2},
    rhs::NDArray,
    i::AbstractUnitRange{<:Integer},
    j::AbstractUnitRange{<:Integer},
) where {T}
    @boundscheck checkbounds(lhs, i, j)
    return _setindex_slice!(
        lhs, rhs, slice_array(_zero_based_range(i), _zero_based_range(j))
    )
end

@inline function Base.getindex(arr::NDArray{T,2}, ::Colon, j::Integer) where {T}
    @boundscheck checkbounds(arr, :, j)
    return nda_get_slice(
        arr, slice_array((0, size(arr, 1)), _zero_based_index(j))
    )
end

@inline function Base.getindex(arr::NDArray{T,2}, i::Integer, ::Colon) where {T}
    @boundscheck checkbounds(arr, i, :)
    return nda_get_slice(arr, slice_array(_zero_based_index(i)))
end

@inline function Base.getindex(
    arr::NDArray{T,2}, i::AbstractUnitRange{<:Integer}, ::Colon
) where {T}
    @boundscheck checkbounds(arr, i, :)
    return nda_get_slice(
        arr, slice_array(_zero_based_range(i), (0, size(arr, 2)))
    )
end

@inline function Base.getindex(
    arr::NDArray{T,2}, ::Colon, j::AbstractUnitRange{<:Integer}
) where {T}
    @boundscheck checkbounds(arr, :, j)
    return nda_get_slice(
        arr, slice_array((0, size(arr, 1)), _zero_based_range(j))
    )
end

@inline function Base.getindex(
    arr::NDArray{T,2}, i::AbstractUnitRange{<:Integer}, j::Integer
) where {T}
    @boundscheck checkbounds(arr, i, j)
    return nda_get_slice(
        arr, slice_array(_zero_based_range(i), _zero_based_index(j))
    )
end

@inline function Base.getindex(
    arr::NDArray{T,2}, i::Integer, j::AbstractUnitRange{<:Integer}
) where {T}
    @boundscheck checkbounds(arr, i, j)
    return nda_get_slice(
        arr, slice_array(_zero_based_index(i), _zero_based_range(j))
    )
end

@inline function Base.getindex(
    arr::NDArray{T,2},
    i::AbstractUnitRange{<:Integer},
    j::AbstractUnitRange{<:Integer},
) where {T}
    @boundscheck checkbounds(arr, i, j)
    return nda_get_slice(
        arr, slice_array(_zero_based_range(i), _zero_based_range(j))
    )
end

@inline function Base.getindex(
    arr::NDArray, i::AbstractUnitRange{<:Integer}
)
    @boundscheck checkbounds(arr, i)
    return nda_get_slice(arr, slice_array(_zero_based_range(i)))
end

@inline function Base.getindex(
    arr::NDArray{T}, c::Vararg{Colon,N}
) where {T,N}
    @boundscheck checkbounds(arr, c...)
    return Base.copy(arr)
end

@inline function Base.setindex!(
    arr::NDArray{T}, rhs::NDArray{T}, c::Vararg{Colon,N}
) where {T,N}
    @boundscheck checkbounds(arr, c...)
    return Base.copyto!(arr, rhs)
end

@inline function Base.setindex!(
    arr::NDArray{T,2}, val::T, ::Colon, j::Integer
) where {T}
    @boundscheck checkbounds(arr, :, j)
    s = nda_get_slice(
        arr, slice_array((0, size(arr, 1)), _zero_based_index(j))
    )
    nda_fill_array(s, val)
    return destroy!(s)
end

@inline function Base.setindex!(
    arr::NDArray{T,2}, val::T, i::Integer, ::Colon
) where {T}
    @boundscheck checkbounds(arr, i, :)
    s = nda_get_slice(arr, slice_array(_zero_based_index(i)))
    nda_fill_array(s, val)
    return destroy!(s)
end

@inline function Base.fill!(arr::NDArray{T}, val::T) where {T}
    nda_fill_array(arr, val)
    return arr
end

#### INITIALIZATION OF NDARRAYS ####
@doc"""
    cuNumeric.fill(val::T, dims::Dims)
    cuNumeric.fill(val::T, dims::Int...)

Create an `NDArray` filled with the scalar value `val`, with the shape specified by `dims`.

# Examples
```@repl
cuNumeric.fill(7.5, (2, 3))
cuNumeric.fill(0, 4)
```
"""
function fill(val::T, dims::Dims) where {T<:SUPPORTED_TYPES}
    return nda_full_array(dims, val)
end

function fill(val::T, dims::Int...) where {T<:SUPPORTED_TYPES}
    return fill(val, dims)
end

function fill(val::T, dim::Int) where {T<:SUPPORTED_TYPES}
    return fill(val, (dim,))
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
trues(dim::Int) = cuNumeric.fill(true, dim)
trues(dims::Dims) = cuNumeric.fill(true, dims)
trues(dims::Int...) = cuNumeric.fill(true, dims)

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
falses(dims::Dims) = cuNumeric.fill(false, dims)
falses(dims::Int...) = cuNumeric.fill(false, dims)
falses(dim::Int) = cuNumeric.fill(false, dim)

@doc"""
    cuNumeric.zeros([T=Float32,] dims::Int...)
    cuNumeric.zeros([T=Float32,] dims::Tuple)

Create an NDArray with element type `T`, of all zeros with size specified by `dims`.
The default type is Float32 if not specified.

# Examples
```@repl
cuNumeric.zeros(2, 2)
cuNumeric.zeros(Float64, 3)
cuNumeric.zeros(Int32, (2,3))
```
"""
function zeros(::Type{T}, dims::Dims{N}) where {T<:SUPPORTED_TYPES,N}
    return nda_zeros_array(dims, T)
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
    return nda_zeros_array((), T)
end

function zeros()
    return zeros(DEFAULT_FLOAT)
end

function zeros_like(arr::NDArray{T,N}) where {T,N}
    return zeros(T, Base.size(arr))
end

@doc"""
    cuNumeric.ones([T=Float32,] dims::Int...)
    cuNumeric.ones([T=Float32,] dims::Tuple)

Create an NDArray with element type `T`, of all zeros with size specified by `dims`.
The default type is Float32 if not specified.

# Examples
```@repl
cuNumeric.ones(2, 2)
cuNumeric.ones(Float32, 3)
cuNumeric.ones(Int32, (2, 3))
```
"""
function ones(::Type{T}, dims::Dims) where {T}
    return nda_full_array(dims, T(1))
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
    return cuNumeric.fill(T(1), ())
end

function ones()
    return ones(DEFAULT_FLOAT)
end

#### OPERATIONS ####
@doc"""
    reshape(arr::NDArray, dims::Dims{N}; copy::Val{C}=Val(false)) where {N,C}
    reshape(arr::NDArray, dims::Int...; copy::Val{C}=Val(false)) where {C}

Return a new `NDArray` reshaped to the specified dimensions.

By default (`copy=Val(false)`) the result shares data with `arr`.
Pass `copy=Val(true)` to allocate a deep copy; the intermediate reshape
view is then destroyed eagerly. Use `Val` (not a runtime `Bool`) so the
return type stays concrete — a `Bool` branch widens inference.

# Examples
```@repl
arr = cuNumeric.ones(4, 3)
reshape(arr, (3, 4))
reshape(arr, 12)
reshape(arr, (3, 4); copy=Val(true))
```
"""

# `copy` is a type parameter via Val{C}, so the default path constant-folds
# and stays type-stable (needed by solve's 1D-rhs reshape).
function reshape(arr::NDArray, i::Dims{N}; copy::Val{C}=Val(false)) where {N,C}
    reshaped = nda_reshape_array(arr, i)
    if C
        copied = Base.copy(reshaped)
        destroy!(reshaped)
        return copied
    end
    return reshaped
end

function reshape(arr::NDArray, i::Int...; copy::Val{C}=Val(false)) where {C}
    return reshape(arr, i; copy=Val{C}())
end

# Ignore the scalar indexing here...
Base.only(x::NDArray{T,0}) where {T} = @allowscalar x[]

function Base.only(x::NDArray{T,N}) where {T,N}
    length(x) == 1 ||
        throw(ArgumentError("collection must contain exactly 1 element"))

    return @allowscalar x[firstindex(x)]
end

unwrap(x::NDArray) = only(x)

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

"""
    h5write(path::String, dataset::String, arr::NDArray)

Write an `NDArray` directly to an HDF5 dataset without a host copy or dimension flip.

# Arguments
- `path`: Path to the HDF5 file.
- `dataset`: Name of the dataset to write.
- `arr`: The array to write.
"""
function h5write(path::String, dataset::String, arr::NDArray{T,N}) where {T,N}
    st_handle = get_store(arr)
    # NDArrays are row-major, so this writes straight through (no dim flip, no warning).
    la = Legate.LogicalArray{T,N}(st_handle, size(arr))
    return Legate.h5write(path, dataset, la)
end

"""
    h5read(path::String, dataset::String; layout::Symbol=:row) -> NDArray

Read a dataset from an HDF5 file into an `NDArray`.

# Arguments
- `path`: Path to the HDF5 file.
- `dataset`: Name of the dataset to read.

# Keywords
- `layout`: On-disk memory order, either `:row` (default) or `:col`.
"""
function h5read(path::String, dataset::String; kwargs...)
    la = Legate.h5read(path, dataset; kwargs...)
    T = eltype(la)
    N = Int(Legate.dim(la))
    st = Legate.data(la.handle)  # call data on the raw impl
    ptr = nda_store_to_ndarray(st)  # pass directly
    arr = NDArray(ptr, T, Val(N), nothing)
    return la.order === :col && N > 1 ? transpose(arr) : arr
end
