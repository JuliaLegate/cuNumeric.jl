export test_factories
export NDArray

lib = "libcwrapper.so"
libnda = joinpath(@__DIR__, "../", "wrapper", "build", lib)

struct Slice
    has_start::Cint
    start::Int64
    has_stop::Cint
    stop::Int64
end

# Opaque pointer
const NDArray_t = Ptr{Cvoid}

# destroy
nda_destroy_array(arr::NDArray_t) = ccall((:nda_destroy_array, libnda),
    Cvoid, (NDArray_t,), arr)

mutable struct NDArray
    ptr::NDArray_t
    function NDArray(ptr)
        handle = new(ptr)
        finalizer(handle) do h
            cuNumeric.nda_destroy_array(h.ptr)
        end
        return handle
    end
end

# construction 
function nda_zeros_array(shape::Vector{UInt64}; type::Union{Nothing,Type{T}}=nothing) where {T}
    dim = Int32(length(shape))
    legate_type = Legate.to_legate_type(isnothing(type) ? Float64 : type)
    ptr = ccall((:nda_zeros_array, libnda),
        NDArray_t, (Int32, Ptr{UInt64}, Legate.LegateTypeAllocated),
        dim, shape, legate_type)
    return NDArray(ptr)
end

function nda_full_array(shape::Vector{UInt64}, value::T) where {T}
    dim = Int32(length(shape))
    type = Legate.to_legate_type(T)
    val = Ref(value)

    ptr = ccall((:nda_full_array, libnda),
        NDArray_t,
        (Int32, Ptr{UInt64}, Legate.LegateTypeAllocated, Ptr{Cvoid}),
        dim, shape, type, val)

    return NDArray(ptr)
end

function nda_random(arr::NDArray, gen_code)
    ccall((:nda_random, libnda),
        Cvoid, (NDArray_t, Int32),
        arr.ptr, Int32(gen_code))
end

function nda_random_array(shape::Vector{UInt64})
    dim = Int32(length(shape))
    ptr = ccall((:nda_random_array, libnda),
        NDArray_t, (Int32, Ptr{UInt64}),
        dim, shape)
    return NDArray(ptr)
end

function nda_get_slice(arr::NDArray, slices::Vector{Slice})
    ptr = ccall((:cn_get_slice, libnda),
        NDArray_t, (NDArray_t, Ptr{Slice}, Cint),
        arr.ptr, pointer(slices), length(slices))
    return NDArray(ptr)
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
function nda_reshape_array(arr::NDArray, newshape::Vector{UInt64})
    dim = Int32(length(newshape))
    ptr = ccall((:nda_reshape_array, libnda),
        NDArray_t, (NDArray_t, Int32, Ptr{UInt64}),
        arr.ptr, dim, newshape)
    return NDArray(ptr)
end

function nda_astype(arr::NDArray, t::Type{T}) where {T}
    type = Legate.to_legate_type(t)

    ptr = ccall((:nda_full_array, libnda),
        NDArray_t,
        (NDArray_t, Legate.LegateTypeAllocated),
        arr.ptr, type)
    return NDArray(ptr)
end

function nda_fill_array(arr::NDArray, value::T) where {T}
    type = Legate.to_legate_type(T)
    val = Ref(value)
    ccall((:nda_fill_array, libnda),
        Cvoid, (NDArray_t, Legate.LegateTypeAllocated, Ptr{Cvoid}),
        arr.ptr, type, val)
    return nothing
end

function nda_assign(arr::NDArray, other::NDArray)
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
        NDArray_t, (NDArray_t, BinaryOpCode, NDArray_t, NDArray_t),
        out.ptr, op_code, rhs1.ptr, rhs2.ptr)
    return out
end

function nda_unary_op(out::NDArray, op_code::UnaryOpCode, input::NDArray)
    ccall((:nda_unary_op, libnda),
        NDArray_t, (NDArray_t, UnaryOpCode, NDArray_t),
        out.ptr, op_code, input.ptr)
    return out
end

function nda_unary_reduction(out::NDArray, op_code::UnaryRedCode, input::NDArray)
    ccall((:nda_unary_reduction, libnda),
        NDArray_t, (NDArray_t, UnaryRedCode, NDArray_t),
        out.ptr, op_code, input.ptr)
    return out
end

function nda_multiply(rhs1::NDArray, rhs2::NDArray, out::NDArray)
    ccall((:nda_multiply, libnda),
        NDArray_t, (NDArray_t, NDArray_t, NDArray_t),
        rhs1.ptr, rhs2.ptr, out.ptr)
    return out
end

function nda_add(rhs1::NDArray, rhs2::NDArray, out::NDArray)
    ccall((:nda_add, libnda),
        NDArray_t, (NDArray_t, NDArray_t, NDArray_t),
        rhs1.ptr, rhs2.ptr, out.ptr)
    return out
end

function nda_multiply_scalar(rhs1::NDArray, value::T) where {T}
    type = Legate.to_legate_type(T)
    val = Ref(value)

    ptr = ccall((:nda_multiply, libnda),
        NDArray_t, (NDArray_t, Legate.LegateTypeAllocated, Ptr{Cvoid}),
        rhs1.ptr, type, val)
    return NDArray(ptr)
end

function nda_add_scalar(rhs1::NDArray, value::T) where {T}
    type = Legate.to_legate_type(T)
    val = Ref(value)

    ptr = ccall((:nda_add, libnda),
        NDArray_t, (NDArray_t, Legate.LegateTypeAllocated, Ptr{Cvoid}),
        rhs1.ptr, type, val)
    return NDArray(ptr)
end

function nda_three_dot_arg(rhs1::NDArray, rhs2::NDArray, out::NDArray)
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

# --- quick smoke test ---
function test_factories()
    shp = UInt64[4, 5, 6]
    a = nda_zeros_array(shp)                 # default type
    b = nda_zeros_array(shp; type=Float64)
    c = nda_full_array(shp, 3.1415)
    for x in (a, b, c)
        @show nda_array_dim(x), nda_array_size(x),
        nda_array_type(x), nda_array_shape(x)
        nda_destroy_array(x)
    end
end
