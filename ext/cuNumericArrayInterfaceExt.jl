module cuNumericArrayInterfaceExt

using ArrayInterface: ArrayInterface
using cuNumeric: cuNumeric, NDArray, @allowscalar
using LinearAlgebra: LinearAlgebra, Adjoint, Transpose

const NDArrayOrWrapper = Union{
    NDArray,
    Adjoint{<:Any,<:NDArray},
    Transpose{<:Any,<:NDArray},
}

ArrayInterface.fast_scalar_indexing(::Type{<:NDArrayOrWrapper}) = false

@inline ArrayInterface.allowed_getindex(x::NDArrayOrWrapper, i...) = @allowscalar(x[i...])
@inline ArrayInterface.allowed_setindex!(x::NDArrayOrWrapper, v, i...) = @allowscalar(x[i...]=v)

# copyto! preserves Julia's linear element order across rank changes, unlike
# cuNumeric.reshape, which follows the backend's C-order reshape convention.
function ArrayInterface.restructure(x::NDArray, y::Union{NDArray,Array})
    return copyto!(similar(x, eltype(y)), y)
end

# Avoid the generic vec/outer-product path, including reading the input values.
function ArrayInterface.zeromatrix(x::NDArray)
    return cuNumeric.zeros(eltype(x), length(x), length(x))
end

# Reject unsupported pivoting before initializing a solver cache.
function ArrayInterface.qr_instance(A::NDArray{<:Any,2}, pivot::LinearAlgebra.ColumnNorm)
    return LinearAlgebra.qr(A, pivot)
end

# Match qr(A)'s concrete type without reading or factoring A.
function ArrayInterface.qr_instance(
    A::NDArray{T,2}, ::LinearAlgebra.NoPivot=LinearAlgebra.NoPivot()
) where {T<:cuNumeric._QR_ACCEPTED}
    S = cuNumeric._qr_eltype(T)
    return cuNumeric.NDArrayQR(similar(A, S, 0, 0), similar(A, S, 0, 0))
end

end
