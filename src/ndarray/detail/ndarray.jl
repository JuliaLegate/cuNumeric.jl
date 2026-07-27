export NDArray
export WrappedNDArray

struct Slice
    has_start::Cint
    start::Int64
    has_stop::Cint
    stop::Int64
end

macro task_scope(scope_name, body)
    TASK_SCOPE_NAMES || return esc(body)
    return quote
        Legate.with_scope($(esc(scope_name))) do
            return $(esc(body))
        end
    end
end

# Opaque pointer
const NDArray_t = Ptr{Cvoid}
const CN_Store_t = Ptr{Cvoid}

# destroy
nda_destroy_array(ptr::NDArray_t) = ccall((:nda_destroy_array, libnda),
    Cvoid, (NDArray_t,), ptr)

nda_nbytes(ptr::NDArray_t) = ccall((:nda_nbytes, libnda),
    Int64, (NDArray_t,), ptr)

function get_julia_type(ptr::NDArray_t)
    type_code = ccall((:nda_array_type_code, libnda), Int32, (NDArray_t,), ptr)
    return Legate.code_type_map[type_code]
end

get_n_dim(ptr::NDArray_t) = Int(ccall((:nda_array_dim, libnda), Int32, (NDArray_t,), ptr))

abstract type AbstractNDArray{T<:SUPPORTED_TYPES,N} end

@doc"""
The NDArray type represents a multi-dimensional array in cuNumeric.
It is a wrapper around a Legate array and provides various methods for array manipulation and operations.
Finalizer calls `nda_destroy_array` to clean up the underlying Legate array when the NDArray is garbage collected.
"""
mutable struct NDArray{T,N,PADDED,P} <: AbstractNDArray{T,N}
    ptr::NDArray_t
    nbytes::Int64
    padding::Union{Nothing,NTuple{N,Int}}
    parent::P

    function NDArray(ptr::NDArray_t, ::Type{T}, ::Val{N}) where {T,N}
        nbytes = cuNumeric.nda_nbytes(ptr)
        cuNumeric.register_alloc!(nbytes)
        handle = new{T,N,false,Nothing}(ptr, nbytes, nothing, nothing)
        finalizer(destroy!, handle)
        return handle
    end

    # Explicit parent inner constructor
    function NDArray(ptr::NDArray_t, ::Type{T}, ::Val{N}, parent::P) where {T,N,P}
        nbytes = cuNumeric.nda_nbytes(ptr)
        cuNumeric.register_alloc!(nbytes)
        handle = new{T,N,false,P}(ptr, nbytes, nothing, parent)
        finalizer(destroy!, handle)
        return handle
    end
end

"""
    destroy!(arr::NDArray)

Eagerly drop the underlying cuPyNumeric/Legate handle and update allocation
counters. Safe to call more than once.
"""
function destroy!(arr::NDArray)
    ptr = arr.ptr
    if ptr != C_NULL
        nbytes = arr.nbytes
        nda_destroy_array(ptr)
        arr.ptr = Ptr{Cvoid}(0)
        arr.nbytes = 0
        nbytes > 0 && register_free!(nbytes)
    end
    return arr
end

# this here is to avoid if else patterns
@inline _NDArray(ptr, T, v, ::Nothing) = NDArray(ptr, T, v)
@inline _NDArray(ptr, T, v, parent) = NDArray(ptr, T, v, parent)

# Dynamic fallback
function NDArray(ptr::NDArray_t; T=get_julia_type(ptr), N::Integer=get_n_dim(ptr), parent=nothing)
    return _NDArray(ptr, T, Val(N), parent)
end

_scope_op(kind, op_code) = string(kind, "#", Int32(op_code))

#! JUST USE FULL TO MAKE a 0D?
# $ cuNumeric.nda_full_array(UInt64[], 2.0f0)
# TODO DAVID TEST THIS HERE
# function NDArray(value::T) where {T <: SUPPORTED_TYPES}
#     type = Legate.to_legate_type(T)
#     ptr = ccall((:nda_from_scalar, libnda),
#         NDArray_t, (Legate.LegateTypeAllocated, Ptr{Cvoid}),
#         type, Ref(value))
#     return NDArray(ptr, T = T, n_dim = 1)
# end

NDArray(value::T) where {T<:SUPPORTED_TYPES} = nda_full_array((), value)

# construction
function nda_zeros_array(dims::Dims{N}, ::Type{T}) where {T,N}
    shape = collect(UInt64, dims)
    legate_type = Legate.to_legate_type(T)
    ptr = @task_scope "zeros" begin
        ccall((:nda_zeros_array, libnda),
            NDArray_t, (Int32, Ptr{UInt64}, Legate.LegateTypeAllocated),
            Int32(N), shape, legate_type)
    end
    return NDArray(ptr, T, Val(N))
end

function nda_full_array(dims::Dims{N}, value::T) where {T,N}
    shape = collect(UInt64, dims)
    type = Legate.to_legate_type(T)

    ptr = @task_scope "full" begin
        ccall((:nda_full_array, libnda),
            NDArray_t,
            (Int32, Ptr{UInt64}, Legate.LegateTypeAllocated, Ptr{Cvoid}),
            Int32(N), shape, type, Ref(value))
    end

    return NDArray(ptr, T, Val(N))
end

function nda_random(arr::NDArray, gen_code)
    @task_scope "rand!" begin
        ccall((:nda_random, libnda),
            Cvoid, (NDArray_t, Int32),
            arr.ptr, Int32(gen_code))
    end
end

function nda_random_array(dims::Dims{N}) where {N}
    shape = collect(UInt64, dims)
    ptr = @task_scope "rand" begin
        ccall((:nda_random_array, libnda),
            NDArray_t, (Int32, Ptr{UInt64}),
            Int32(N), shape)
    end
    return NDArray(ptr, Float64, Val(N)) #* T is always Float64 cause of cupynumeric
end

function nda_get_slice(arr::NDArray{T,N}, slices::Vector{Slice}) where {T,N}
    ptr = @task_scope "slice" begin
        ccall((:nda_get_slice, libnda),
            NDArray_t, (NDArray_t, Ptr{Slice}, Cint),
            arr.ptr, pointer(slices), length(slices))
    end
    # Keep parent so callers can detect views (slices share the parent store).
    return NDArray(ptr, T, Val(N), arr)
end

# queries
nda_array_dim(arr::NDArray) = ccall((:nda_array_dim, libnda),
    Int32, (NDArray_t,), arr.ptr)
nda_array_size(arr::NDArray) = ccall((:nda_array_size, libnda),
    Int32, (NDArray_t,), arr.ptr)
function nda_array_type_code(arr::NDArray)
    return ccall((:nda_array_type_code, libnda),
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
function nda_reshape_array(arr::NDArray{T}, newdims::Dims{N}) where {T,N}
    newshape = collect(UInt64, newdims)
    ptr = @task_scope "reshape" begin
        ccall((:nda_reshape_array, libnda),
            NDArray_t, (NDArray_t, Int32, Ptr{UInt64}),
            arr.ptr, Int32(N), newshape)
    end
    return NDArray(ptr, T, Val(N))
end

function nda_astype(arr::NDArray{OLD_T,N}, ::Type{NEW_T}) where {OLD_T,NEW_T,N}
    type = Legate.to_legate_type(NEW_T)
    ptr = @task_scope "astype" begin
        ccall((:nda_astype, libnda),
            NDArray_t,
            (NDArray_t, Legate.LegateTypeAllocated),
            arr.ptr, type)
    end
    return NDArray(ptr, NEW_T, Val(N))
end

function nda_fill_array(arr::NDArray{T}, value::T) where {T}
    type = Legate.to_legate_type(T)
    val = Ref(value)
    @task_scope "fill!" begin
        ccall((:nda_fill_array, libnda),
            Cvoid, (NDArray_t, Legate.LegateTypeAllocated, Ptr{Cvoid}),
            arr.ptr, type, val)
    end
    return nothing
end

function nda_assign(arr::NDArray{T}, other::NDArray{T}) where {T}
    @task_scope "copyto!" begin
        ccall((:nda_assign, libnda),
            Cvoid, (NDArray_t, NDArray_t),
            arr.ptr, other.ptr)
    end
end

function nda_copy(arr::NDArray{T,N}) where {T,N}
    ptr = @task_scope "copy" begin
        ccall((:nda_copy, libnda),
            NDArray_t, (NDArray_t,),
            arr.ptr)
    end
    return NDArray(ptr, T, Val(N))
end

# src will be unused after this
function nda_move(dst::NDArray{T,N}, src::NDArray{T,N}) where {T,N}
    @task_scope "move!" begin
        ccall((:nda_move, libnda),
            Cvoid, (NDArray_t, NDArray_t),
            dst.ptr, src.ptr)
    end

    src.ptr = Ptr{Cvoid}(0)
    src.nbytes = 0
    return register_free!(dst.nbytes)
end

# operations
function nda_binary_op!(out::NDArray, op_code::BinaryOpCode, rhs1::NDArray, rhs2::NDArray)
    @task_scope _scope_op("binary", op_code) begin
        ccall((:nda_binary_op, libnda),
            Cvoid, (NDArray_t, BinaryOpCode, NDArray_t, NDArray_t),
            out.ptr, op_code, rhs1.ptr, rhs2.ptr)
    end
    return out
end

function nda_unary_op!(out::NDArray, op_code::UnaryOpCode, input::NDArray)
    @task_scope _scope_op("unary", op_code) begin
        ccall((:nda_unary_op, libnda),
            Cvoid, (NDArray_t, UnaryOpCode, NDArray_t),
            out.ptr, op_code, input.ptr)
    end
    return out
end

function nda_unary_reduction(out::NDArray, op_code::UnaryRedCode, input::NDArray)
    @task_scope _scope_op("reduce", op_code) begin
        ccall((:nda_unary_reduction, libnda),
            Cvoid, (NDArray_t, UnaryRedCode, NDArray_t),
            out.ptr, op_code, input.ptr)
    end
    return out
end

function nda_unary_reduction_axes(
    op_code::UnaryRedCode, input::NDArray{T,N}, axes::Vector{Int32}, keepdims::Bool
) where {T,N}
    axes_c = collect(Int32, axes)
    ptr = @task_scope _scope_op("reduce_axes", op_code) begin
        ccall((:nda_unary_reduction_axes, libnda),
            NDArray_t, (UnaryRedCode, NDArray_t, Ptr{Int32}, Int32, Cint),
            op_code, input.ptr, axes_c, Int32(length(axes_c)), keepdims)
    end
    return NDArray(ptr)
end

function nda_array_equal(rhs1::NDArray{T,N}, rhs2::NDArray{T,N}) where {T,N}
    ptr = @task_scope "array_equal" begin
        ccall((:nda_array_equal, libnda),
            NDArray_t, (NDArray_t, NDArray_t),
            rhs1.ptr, rhs2.ptr)
    end
    return NDArray(ptr, Bool, Val(1))
end

# 2D -> 1D: extract the k-th diagonal. Backend only supports the 2D case
# (1D-construct and >2D both abort), so non-2D input is a MethodError.
function nda_diag(arr::NDArray{T,2}, k::Int32) where {T}
    ptr = @task_scope "diag" begin
        ccall((:nda_diag, libnda),
            NDArray_t, (NDArray_t, Int32),
            arr.ptr, k)
    end
    return NDArray(ptr, T, Val(1))
end

# unique always returns a flat 1D array of the input's element type
function nda_unique(arr::NDArray{T}) where {T}
    ptr = @task_scope "unique" begin
        ccall((:nda_unique, libnda),
            NDArray_t, (NDArray_t,),
            arr.ptr)
    end
    return NDArray(ptr, T, Val(1))
end

function nda_ravel(arr::NDArray)
    ptr = @task_scope "ravel" begin
        ccall((:nda_ravel, libnda),
            NDArray_t, (NDArray_t,),
            arr.ptr)
    end
    return NDArray(ptr)
end

function nda_add(rhs1::NDArray, rhs2::NDArray, out::NDArray)
    @task_scope "add" begin
        ccall((:nda_add, libnda),
            Cvoid, (NDArray_t, NDArray_t, NDArray_t),
            rhs1.ptr, rhs2.ptr, out.ptr)
    end
    return out
end

function nda_multiply_scalar(rhs1::NDArray{T,N}, value::T) where {T,N}
    type = Legate.to_legate_type(T)

    ptr = @task_scope "multiply_scalar" begin
        ccall((:nda_multiply_scalar, libnda),
            NDArray_t, (NDArray_t, Legate.LegateTypeAllocated, Ptr{Cvoid}),
            rhs1.ptr, type, Ref(value))
    end
    return NDArray(ptr, T, Val(N))
end

function nda_add_scalar(rhs1::NDArray{T,N}, value::T) where {T,N}
    type = Legate.to_legate_type(T)

    ptr = @task_scope "add_scalar" begin
        ccall((:nda_add_scalar, libnda),
            NDArray_t, (NDArray_t, Legate.LegateTypeAllocated, Ptr{Cvoid}),
            rhs1.ptr, type, Ref(value))
    end
    return NDArray(ptr, T, Val(N))
end

function nda_three_dot_arg(rhs1::NDArray{T}, rhs2::NDArray{T}, out::NDArray{T}) where {T}
    @task_scope "matmul" begin
        ccall((:nda_three_dot_arg, libnda),
            Cvoid, (NDArray_t, NDArray_t, NDArray_t),
            rhs1.ptr, rhs2.ptr, out.ptr)
    end
    return out
end

function nda_dot(rhs1::NDArray, rhs2::NDArray)
    ptr = @task_scope "dot" begin
        ccall((:nda_dot, libnda),
            NDArray_t, (NDArray_t, NDArray_t),
            rhs1.ptr, rhs2.ptr)
    end
    return NDArray(ptr)
end

function nda_eye(rows::Int32, ::Type{T}) where {T}
    legate_type = Legate.to_legate_type(T)
    ptr = @task_scope "eye" begin
        ccall((:nda_eye, libnda),
            NDArray_t, (Int32, Legate.LegateTypeAllocated),
            rows, legate_type)
    end
    return NDArray(ptr, T, Val(2))
end

function nda_trace(
    arr::NDArray, offset::Int32, a1::Int32, a2::Int32, ::Type{T}
) where {T}
    legate_type = Legate.to_legate_type(T)
    ptr = @task_scope "trace" begin
        ccall((:nda_trace, libnda),
            NDArray_t,
            (NDArray_t, Int32, Int32, Int32, Legate.LegateTypeAllocated),
            arr.ptr, offset, a1, a2, legate_type)
    end
    return NDArray(ptr, T, Val(1))
end

# transpose reverses the axes: element type and rank are preserved
function nda_transpose(arr::NDArray{T,N}) where {T,N}
    ptr = @task_scope "transpose" begin
        ccall((:nda_transpose, libnda),
            NDArray_t, (NDArray_t,),
            arr.ptr)
    end
    return NDArray(ptr, T, Val(N))
end

function nda_attach_external(arr::Array{T,N}; shape::Dims{N}=size(arr)) where {T,N}
    st = Legate.attach_external_row_major(arr; shape)
    # Use the CxxWrap method for type-safe interaction
    # This returns a raw pointer compatible with the NDArray constructor
    # `nda_store_to_ndarray` takes the store by value; drop the Julia-owned
    # LogicalStoreImpl so it does not pin alongside the NDArray until GC.
    nda_ptr = cuNumeric.nda_store_to_ndarray(st.handle)
    finalize(st.handle)
    return NDArray(nda_ptr, T, Val(N), arr)
end

# return underlying logical store to the NDArray obj
function get_store(arr::NDArray)
    cxx_ptr = CxxWrap.CxxPtr{CN_NDArray}(arr.ptr)
    return _get_store(cxx_ptr)
end

function get_ptr(arr::NDArray{T,N}) where {T,N}
    # `get_store` returns a Julia-owned LogicalArrayImplAllocated that shares the
    # store with the NDArray; finalize after use (same pin class as `_add_task_array!`).
    st_handle = get_store(arr) # LogicalArrayImplAllocated (returned by value)
    la = Legate.LogicalArray{T,N}(st_handle, size(arr))
    ptr = Legate.get_ptr(la)
    finalize(st_handle)
    return ptr
end

@doc"""
    to_cpp_index(idx::Dims{N}, ::Type{T}=UInt64) where {N}

**Internal API**

Converts a Julia 1-based index tuple `idx` to a zero-based C++ style index wrapped in `StdVector` of the specified integer type.

Each element of `idx` is decremented by 1 to adjust from Julia’s 1-based indexing to C++ 0-based indexing.
"""
function to_cpp_index(idx::Dims{N}, (::Type{T})=UInt64) where {N,T<:Integer}
    return StdVector(T.([e - 1 for e in idx]))
end

@doc"""
    to_cpp_index(d::Int64, ::Type{T}=UInt64)

**Internal API**

Converts a single Julia 1-based index `d` to a zero-based C++ style index wrapped in `StdVector`.
"""
to_cpp_index(d::Int64, (::Type{T})=UInt64) where {T} = StdVector(T.([d - 1]))

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
    return cuNumeric.Slice(
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
shape(arr::NDArray{<:Any,N,true}) where {N} = arr.padding

function shape(arr::NDArray{<:Any,N,false}) where {N}
    shp = cuNumeric.nda_array_shape(arr)
    return ntuple(i -> Int(shp[i]), Val(N))
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
function compare(
    julia_array::AbstractArray{T1,N}, arr::NDArray{T2,N}, atol::Real, rtol::Real
) where {T1,T2,N}
    if (shape(arr) != Base.size(julia_array))
        @warn "NDArray has shape $(shape(arr)) and Julia array has shape $(Base.size(julia_array))!\n"
        return false
    end

    for CI in CartesianIndices(julia_array)
        x = julia_array[CI];
        y = arr[Tuple(CI)...]
        if !isapprox(x, y; atol=atol, rtol=rtol)
            return false
        end
    end

    # successful completion
    return true
end

function compare(
    arr::NDArray{T2,N}, julia_array::AbstractArray{T1,N}, atol::Real, rtol::Real
) where {T1,T2,N}
    return compare(julia_array, arr, atol, rtol)
end

function compare(arr::NDArray{T,N}, arr2::NDArray{T,N}, atol::Real, rtol::Real) where {T,N}
    if (shape(arr) != shape(arr2))
        @warn "NDArray LHS has shape $(shape(arr)) and NDArray RHS has shape $(shape(arr2))!\n"
        return false
    end

    dims = shape(arr)
    for CI in CartesianIndices(dims)
        x = arr[Tuple(CI)...];
        y = arr2[Tuple(CI)...]
        if !isapprox(x, y; atol=atol, rtol=rtol)
            return false
        end
    end

    # successful completion
    return true
end

function nda_to_logical_store(arr::NDArray{T,N}) where {T,N}
    la_handle = cuNumeric.get_store(arr) # LogicalArrayImplAllocated (returned by value)
    st_handle = Legate.data(Legate.LogicalArray{T,N}(la_handle, size(arr)))
    # Drop temp LogicalArray owner after extracting the store.
    finalize(la_handle)
    return Legate.LogicalStore{T,N}(st_handle, size(arr))
end

function nda_to_logical_array(arr::NDArray{T,N}) where {T,N}
    st_handle = cuNumeric.get_store(arr)
    return Legate.LogicalArray{T,N}(st_handle, size(arr))
end
