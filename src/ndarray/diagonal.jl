@doc"""
    cuNumeric.diag(arr::NDArray; k=0)

Extract the k-th diagonal from a 2D `NDArray`.
"""
function diag(arr::NDArray; k::Int=0)
    return nda_diag(arr, Int32(k))
end

@doc"""
    cuNumeric._eye([T=Float32,] rows::Int)

Create a 2D identity `NDArray` of size `rows × rows` with element type `T`.
The default type is Float32 if not specified. Prefer LinearAlgebra.I instead
    or `LinearAlgebra.Diagonal`
"""
function _eye(::Type{T}, rows::Int) where {T}
    return nda_eye(Int32(rows), T)
end
function _eye(rows::Int)
    return eye(DEFAULT_FLOAT, rows)
end

@doc"""
    cuNumeric.trace(arr::NDArray; offset=0, a1=0, a2=1)

Compute the trace (sum of a diagonal) of the `NDArray`.
The accumulator type follows promotions of other reductions like 'sum'.
"""
function trace(arr::NDArray{T}; offset::Int=0, a1::Int=0, a2::Int=1) where {T}
    T_OUT = Base.promote_op(Base.sum, Vector{T})
    return nda_trace(arr, Int32(offset), Int32(a1), Int32(a2), T_OUT)
end

# Prefer the cuNumeric backend over LinearAlgebra.diag's scalar-indexing fallback
# now that NDArray <: AbstractArray.
LinearAlgebra.diag(arr::NDArray{<:Any,2}, k::Integer=0) = nda_diag(arr, Int32(k))

# Wrap without copying. Must call the typed inner constructor — `Diagonal(arr)`
# would recurse into this method instead of LinearAlgebra's AbstractVector method.
function LinearAlgebra.Diagonal(arr::NDArray{T,1}) where {T}
    return Diagonal{T,typeof(arr)}(arr)
end

function LinearAlgebra.Diagonal(arr::NDArray{T,2}) where {T}
    return Diagonal(diag(arr))
end

function Base.show(
    io::IO,
    D::LinearAlgebra.Diagonal{T,V},
) where {T,V<:NDArray{T,1}}
    host_D = LinearAlgebra.Diagonal(Array(D.diag))
    return show(io, host_D)
end

function Base.show(
    io::IO,
    ::MIME"text/plain",
    D::LinearAlgebra.Diagonal{T,V},
) where {T,V<:NDArray{T,1}}
    summary(io, D)
    isempty(D) && return nothing

    println(io, ":")
    host_D = LinearAlgebra.Diagonal(Array(D.diag))
    return Base.print_array(io, host_D)
end

#### UniformScaling (LinearAlgebra.I) ####

# Dense λI with element type R. Avoids LinearAlgebra's scalar-indexing fallbacks.
@inline function _uniformscaling_eye(::Type{R}, n::Integer, λ) where {R}
    E = _eye(R, Int(n))
    return isone(λ) ? E : nda_multiply_scalar(E, R(λ))
end

# Like CuArray(I, n, n) / CuArray{T}(I, dims)
function NDArray{T}(J::LinearAlgebra.UniformScaling, dims::Dims{2}) where {T}
    A = zeros(T, dims)
    copyto!(A, J)
    return A
end
function NDArray{T}(J::LinearAlgebra.UniformScaling, m::Integer, n::Integer) where {T}
    NDArray{T}(J, Dims((Int(m), Int(n))))
end
NDArray(J::LinearAlgebra.UniformScaling{T}, dims::Dims{2}) where {T} = NDArray{T}(J, dims)
function NDArray(J::LinearAlgebra.UniformScaling{T}, m::Integer, n::Integer) where {T}
    NDArray{T}(J, Dims((Int(m), Int(n))))
end

function Base.copyto!(A::NDArray{T,2}, J::LinearAlgebra.UniformScaling) where {T}
    m, n = size(A)
    if iszero(J.λ)
        return fill!(A, zero(T))
    elseif m == n
        return copyto!(A, _uniformscaling_eye(T, m, J.λ))
    else
        fill!(A, zero(T))
        k = min(m, n)
        A[1:k, 1:k] = _uniformscaling_eye(T, k, J.λ)
        return A
    end
end

function Base.:+(A::NDArray{T,2}, J::LinearAlgebra.UniformScaling) where {T}
    LinearAlgebra.checksquare(A)
    R = Base.promote_op(+, T, typeof(J.λ))
    return A + _uniformscaling_eye(R, size(A, 1), J.λ)
end
Base.:+(J::LinearAlgebra.UniformScaling, A::NDArray{<:Any,2}) = A + J

Base.:-(A::NDArray{<:Any,2}, J::LinearAlgebra.UniformScaling) = A + (-J)
function Base.:-(J::LinearAlgebra.UniformScaling, A::NDArray{<:Any,2})
    return (-A) + J
end

# Scale by λ without promoting the array (A * I must not Bool→Float32 promote).
function Base.:*(A::NDArray{T}, J::LinearAlgebra.UniformScaling) where {T}
    return _mul_scalar(T, J.λ, A)
end
function Base.:*(J::LinearAlgebra.UniformScaling, A::NDArray{T}) where {T}
    return _mul_scalar(T, J.λ, A)
end

function Base.one(A::NDArray{T,2}) where {T}
    LinearAlgebra.checksquare(A)
    return eye(T, size(A, 1))
end
function Base.oneunit(A::NDArray{T,2}) where {T}
    LinearAlgebra.checksquare(A)
    return eye(T, size(A, 1))
end
