export NDArray

struct Slice
    has_start::Cint
    start::Int64
    has_stop::Cint
    stop::Int64
end

# Opaque pointer
const NDArray_t = Ptr{Cvoid}

# destroy
nda_destroy_array(ptr::NDArray_t) = ccall((:nda_destroy_array, libnda),
    Cvoid, (NDArray_t,), ptr)

nda_nbytes(ptr::NDArray_t) = ccall((:nda_nbytes, libnda),
    Int64, (NDArray_t,), ptr)

function get_julia_type(ptr::NDArray_t) 
    type_code = ccall((:nda_array_type_code, libnda), Int32, (NDArray_t,), ptr)
    return Legate.code_type_map[type_code]
end

get_n_dim(ptr::NDArray_t) = ccall((:nda_array_dim, libnda), Int32, (NDArray_t,), ptr)

@doc"""
**Internal API**

The NDArray type represents a multi-dimensional array in cuNumeric.
It is a wrapper around a Legate array and provides various methods for array manipulation and operations. 
Finalizer calls `nda_destroy_array` to clean up the underlying Legate array when the NDArray is garbage collected.
"""
mutable struct NDArray{T,N}
    ptr::NDArray_t
    nbytes::Int64
    function NDArray(ptr::NDArray_t; T = get_julia_type(ptr), n_dim = get_n_dim(ptr))
        nbytes = cuNumeric.nda_nbytes(ptr)
        cuNumeric.register_alloc!(nbytes)
        handle = new{T, n_dim}(ptr, nbytes)
        finalizer(handle) do h
            cuNumeric.nda_destroy_array(h.ptr)
            cuNumeric.register_free!(h.nbytes)
        end
        return handle
    end
end

#* SHOULD THE DIM ON THIS BE 0??
function NDArray(value::T) where {T <: SUPPORTED_TYPES}
    type = Legate.to_legate_type(T)
    ptr = ccall((:nda_from_scalar, libnda),
        NDArray_t, (Legate.LegateTypeAllocated, Ptr{Cvoid}),
        type, Ref(value))
    return NDArray(ptr, T = T, n_dim = 1)
end

# construction 
function nda_zeros_array(shape::Vector{UInt64}, ::Type{T}) where {T}
    n_dim = Int32(length(shape))
    legate_type = Legate.to_legate_type(T)
    ptr = ccall((:nda_zeros_array, libnda),
        NDArray_t, (Int32, Ptr{UInt64}, Legate.LegateTypeAllocated),
        n_dim, shape, legate_type)
    return NDArray(ptr; T = T, n_dim = n_dim)
end

function nda_full_array(shape::Vector{UInt64}, value::T) where {T}
    n_dim = Int32(length(shape))
    type = Legate.to_legate_type(T)

    ptr = ccall((:nda_full_array, libnda),
        NDArray_t,
        (Int32, Ptr{UInt64}, Legate.LegateTypeAllocated, Ptr{Cvoid}),
        n_dim, shape, type, Ref(value))

    return NDArray(ptr; T = T, n_dim = n_dim)
end

function nda_random(arr::NDArray, gen_code)
    ccall((:nda_random, libnda),
        Cvoid, (NDArray_t, Int32),
        arr.ptr, Int32(gen_code))
end

function nda_random_array(shape::Vector{UInt64})
    n_dim = Int32(length(shape))
    ptr = ccall((:nda_random_array, libnda),
        NDArray_t, (Int32, Ptr{UInt64}),
        n_dim, shape)
    return NDArray(ptr; n_dim = n_dim)
end

function nda_get_slice(arr::NDArray{T,N}, slices::Vector{Slice}) where {T,N}
    ptr = ccall((:nda_get_slice, libnda),
        NDArray_t, (NDArray_t, Ptr{Slice}, Cint),
        arr.ptr, pointer(slices), length(slices))
    return NDArray(ptr; T = T, n_dim = N)
end

# queries
nda_array_dim(arr::NDArray) = ccall((:nda_array_dim, libnda),
    Int32, (NDArray_t,), arr.ptr)
nda_array_size(arr::NDArray) = ccall((:nda_array_size, libnda),
    Int32, (NDArray_t,), arr.ptr)
function nda_array_type_code(arr::NDArray)
    ccall((:nda_array_type_code, libnda),
        Int32, (NDArray_t,), arr.ptr)
end

function nda_array_shape(arr::NDArray)
    d = Int(nda_array_dim(arr))
    buf = Vector{UInt64}(undef, d)
    ccall((:nda_array_shape, libnda),
        Cvoid, (NDArray_t, Ptr{UInt64}),
        arr.ptr, buf)
    return buf
end

# modify
function nda_reshape_array(arr::NDArray{T}, newshape::Vector{UInt64}) where T
    n_dim = Int32(length(newshape))
    ptr = ccall((:nda_reshape_array, libnda),
        NDArray_t, (NDArray_t, Int32, Ptr{UInt64}),
        arr.ptr, n_dim, newshape)
    return NDArray(ptr; T = T, n_dim = n_dim)
end

function nda_astype(arr::NDArray{OLD_T, N}, ::Type{NEW_T}) where {OLD_T, NEW_T, N}
    type = Legate.to_legate_type(NEW_T)
    ptr = ccall((:nda_astype, libnda),
        NDArray_t,
        (NDArray_t, Legate.LegateTypeAllocated),
        arr.ptr, type)
    return NDArray(ptr; T = NEW_T, n_dim = N)
end

function nda_fill_array(arr::NDArray{T}, value::T) where {T}
    type = Legate.to_legate_type(T)
    val = Ref(value)
    ccall((:nda_fill_array, libnda),
        Cvoid, (NDArray_t, Legate.LegateTypeAllocated, Ptr{Cvoid}),
        arr.ptr, type, val)
    return nothing
end

#! probably should be copyto!
function nda_assign(arr::NDArray{T}, other::NDArray{T}) where T
    ccall((:nda_assign, libnda),
        Cvoid, (NDArray_t, NDArray_t),
        arr.ptr, other.ptr)
end

function nda_copy(arr::NDArray)
    ptr = ccall((:nda_copy, libnda),
        NDArray_t, (NDArray_t,),
        arr.ptr)
    return NDArray(ptr)
end

# operations 
function nda_binary_op(out::NDArray, op_code::BinaryOpCode, rhs1::NDArray, rhs2::NDArray)
    ccall((:nda_binary_op, libnda),
        Cvoid, (NDArray_t, BinaryOpCode, NDArray_t, NDArray_t),
        out.ptr, op_code, rhs1.ptr, rhs2.ptr)
    return out
end

function nda_unary_op(out::NDArray, op_code::UnaryOpCode, input::NDArray)
    ccall((:nda_unary_op, libnda),
        Cvoid, (NDArray_t, UnaryOpCode, NDArray_t),
        out.ptr, op_code, input.ptr)
    return out
end

function nda_unary_reduction(out::NDArray, op_code::UnaryRedCode, input::NDArray)
    ccall((:nda_unary_reduction, libnda),
        Cvoid, (NDArray_t, UnaryRedCode, NDArray_t),
        out.ptr, op_code, input.ptr)
    return out
end

function nda_multiply(rhs1::NDArray, rhs2::NDArray, out::NDArray)
    ccall((:nda_multiply, libnda),
        Cvoid, (NDArray_t, NDArray_t, NDArray_t),
        rhs1.ptr, rhs2.ptr, out.ptr)
    return out
end

function nda_add(rhs1::NDArray, rhs2::NDArray, out::NDArray)
    ccall((:nda_add, libnda),
        Cvoid, (NDArray_t, NDArray_t, NDArray_t),
        rhs1.ptr, rhs2.ptr, out.ptr)
    return out
end

function nda_multiply_scalar(rhs1::NDArray{T, N}, value::T) where {T, N}
    type = Legate.to_legate_type(T)

    ptr = ccall((:nda_multiply_scalar, libnda),
        NDArray_t, (NDArray_t, Legate.LegateTypeAllocated, Ptr{Cvoid}),
        rhs1.ptr, type, Ref(value))
    return NDArray(ptr; T = T, n_dim = N)
end

function nda_add_scalar(rhs1::NDArray{T, N}, value::T) where {T, N}
    type = Legate.to_legate_type(T)

    ptr = ccall((:nda_add_scalar, libnda),
        NDArray_t, (NDArray_t, Legate.LegateTypeAllocated, Ptr{Cvoid}),
        rhs1.ptr, type, Ref(value))
    return NDArray(ptr; T = T, n_dim = N)
end

function nda_three_dot_arg(rhs1::NDArray{T}, rhs2::NDArray{T}, out::NDArray{T}) where T
    ccall((:nda_three_dot_arg, libnda),
        Cvoid, (NDArray_t, NDArray_t, NDArray_t),
        rhs1.ptr, rhs2.ptr, out.ptr)
    return out
end

function nda_dot(rhs1::NDArray, rhs2::NDArray)
    ptr = ccall((:nda_dot, libnda),
        NDArray_t, (NDArray_t, NDArray_t),
        rhs1.ptr, rhs2.ptr)
    return NDArray(ptr)
end

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

**Internal API**

Returns the element type of the `NDArray`.

This method uses `nda_array_type_code` internally to map to the appropriate Julia element type.
"""
Base.eltype(arr::NDArray) = Legate.code_type_map[nda_array_type_code(arr)]

@doc"""
    LegateType(T::Type)

**Internal API**

Converts a Julia type `T` to the corresponding Legate type.
"""
LegateType(T::Type) = Legate.to_legate_type(T)

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
    shape(arr::NDArray)

**Internal API**

Return the size of the given `NDArray`.
"""
shape(arr::NDArray) = Tuple(Int.(cuNumeric.nda_array_shape(arr)))

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

!!! warning

    This function uses scalar indexing and should not be used in production code. This is meant for testing.


# Notes
- This is an internal API used by higher-level approximate equality functions.
- Does not support relative tolerance (`rtol`).

# Behavior
- Checks size compatibility.
- Checks element type compatibility for `NDArray` vs Julia array.
- Iterates over elements using `CartesianIndices` to compare element-wise difference.
"""
function compare(julia_array::AbstractArray, arr::NDArray, atol::Real, rtol::Real)
    if (shape(arr) != Base.size(julia_array))
        @warn "NDArray has shape $(shape(arr)) and Julia array has shape $(Base.size(julia_array))!\n"
        return false
    end

    if (eltype(arr) != eltype(julia_array))
        @warn "NDArray has eltype $(eltype(arr)) and Julia array has eltype $(eltype(julia_array))!\n"
        return false
    end

    for CI in CartesianIndices(julia_array)
        x = julia_array[CI]; y = arr[Tuple(CI)...]
        if !isapprox(x, y; atol = atol, rtol = rtol)
            return false
        end
    end

    # successful completion
    return true
end

function compare(arr::NDArray, julia_array::AbstractArray, atol::Real, rtol::Real)
    return compare(julia_array, arr, atol, rtol)
end

function compare(arr::NDArray, arr2::NDArray, atol::Real, rtol::Real)
    if (shape(arr) != shape(arr2))
        @warn "NDArray LHS has shape $(shape(arr)) and NDArray RHS has shape $(shape(arr2))!\n"
        return false
    end

    dims = shape(arr)
    for CI in CartesianIndices(dims)
        x = arr[Tuple(CI)...]; y = arr2[Tuple(CI)...]
        if !isapprox(x, y; atol = atol, rtol = rtol)
            return false
        end
    end

    # successful completion
    return true
end
