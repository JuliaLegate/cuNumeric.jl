###### diag / _eye / trace ######

@doc"""
    cuNumeric.diag(arr::NDArray; k=0)

Extract the k-th diagonal from a 2D `NDArray`.
"""
function diag(arr::NDArray; k::Int=0)
    return nda_diag(arr, Int32(k))
end

LinearAlgebra.diag(arr::NDArray{<:Any,2}, k::Integer=0) = nda_diag(arr, Int32(k))

# Internal dense identity used by UniformScaling / Diagonal densify helpers.
# Prefer `LinearAlgebra.I` / `NDArray{T}(I, n, n)` / `one(A)` in user code.
function _eye(::Type{T}, rows::Int) where {T}
    return nda_eye(Int32(rows), T)
end
_eye(rows::Int) = _eye(DEFAULT_FLOAT, rows)

@doc"""
    cuNumeric.trace(arr::NDArray; offset=0, a1=0, a2=1)

Compute the trace (sum of a diagonal) of the `NDArray`.
The accumulator type follows promotions of other reductions like 'sum'.
"""
function trace(arr::NDArray{T}; offset::Int=0, a1::Int=0, a2::Int=1) where {T}
    T_OUT = Base.promote_op(Base.sum, Vector{T})
    return nda_trace(arr, Int32(offset), Int32(a1), Int32(a2), T_OUT)
end

function LinearAlgebra.tr(arr::NDArray{<:Any,2})
    T_OUT = Base.promote_op(Base.sum, Vector{T})
    return nda_trace(arr, Int32(0), Int32(0), Int32(1), T_OUT)
end

###### Diagonal constructors ######

const DiagonalNDArray{T} = Diagonal{T,<:NDArray{T,1}}

# Wrap without copying. Must call the typed inner constructor — `Diagonal(arr)`
# would recurse into this method instead of LinearAlgebra's AbstractVector method.
function LinearAlgebra.Diagonal(arr::NDArray{T,1}) where {T}
    return Diagonal{T,typeof(arr)}(arr)
end

function LinearAlgebra.Diagonal(arr::NDArray{T,2}) where {T}
    return Diagonal(diag(arr))
end

# Note: Matrix{T} === Array{T,2}, so do not also define Array{T,2}(...).
# Use Base.zeros — bare `zeros` resolves to cuNumeric.zeros inside this module.
function Base.Matrix{T}(D::DiagonalNDArray) where {T}
    dv = Array(D.diag)
    n = length(dv)
    B = Base.zeros(T, n, n)
    @inbounds for i in 1:n
        B[i, i] = dv[i]
    end
    return B
end
Base.Matrix(D::DiagonalNDArray{T}) where {T} = Matrix{T}(D)

function Base.show(io::IO, D::DiagonalNDArray)
    show(io, Diagonal(Array(D.diag)))
end

function Base.show(io::IO, ::MIME"text/plain", D::DiagonalNDArray)
    # Keep the real Diagonal{T,<:NDArray} in the summary; only densify the
    # diagonal vector for Base's ⋅-style body formatting.
    summary(io, D)
    isempty(D) && return nothing
    println(io, ":")
    Base.print_array(io, Diagonal(Array(D.diag)))
    return nothing
end

###### Diagonal operators ######

@inline _diag_vec(D::DiagonalNDArray) = D.diag
@inline _row_scale(d::NDArray{<:Any,1}) = reshape(d, (length(d), 1))
@inline _col_scale(d::NDArray{<:Any,1}) = reshape(d, (1, length(d)))

function Base.:*(D::DiagonalNDArray, A::NDArray{<:Any,2})
    size(A, 1) == size(D, 1) || throw(
        DimensionMismatch(
            "matrix is $(size(A,1))×$(size(A,2)), but diagonal is $(size(D,1))×$(size(D,2))"
        ),
    )
    return _row_scale(_diag_vec(D)) .* A
end

function Base.:*(A::NDArray{<:Any,2}, D::DiagonalNDArray)
    size(A, 2) == size(D, 1) || throw(
        DimensionMismatch(
            "matrix is $(size(A,1))×$(size(A,2)), but diagonal is $(size(D,1))×$(size(D,2))"
        ),
    )
    return A .* _col_scale(_diag_vec(D))
end

function Base.:*(D::DiagonalNDArray, v::NDArray{<:Any,1})
    length(v) == size(D, 1) || throw(
        DimensionMismatch("vector length $(length(v)) does not match diagonal $(size(D,1))")
    )
    return _diag_vec(D) .* v
end

function LinearAlgebra.lmul!(D::DiagonalNDArray, B::NDArray)
    return copyto!(B, D * B)
end

function LinearAlgebra.rmul!(A::NDArray, D::DiagonalNDArray)
    return copyto!(A, A * D)
end

function LinearAlgebra.mul!(C::NDArray, D::DiagonalNDArray, A::NDArray)
    return copyto!(C, D * A)
end

function LinearAlgebra.mul!(C::NDArray, A::NDArray, D::DiagonalNDArray)
    return copyto!(C, A * D)
end

function Base.:\(D::DiagonalNDArray, B::NDArray{<:Any,1})
    length(B) == size(D, 1) || throw(
        DimensionMismatch("vector length $(length(B)) does not match diagonal $(size(D,1))")
    )
    return B ./ _diag_vec(D)
end

function Base.:\(D::DiagonalNDArray, B::NDArray{<:Any,2})
    size(B, 1) == size(D, 1) || throw(
        DimensionMismatch(
            "matrix is $(size(B,1))×$(size(B,2)), but diagonal is $(size(D,1))×$(size(D,2))"
        ),
    )
    return B ./ _row_scale(_diag_vec(D))
end

function Base.:/(A::NDArray{<:Any,2}, D::DiagonalNDArray)
    size(A, 2) == size(D, 1) || throw(
        DimensionMismatch(
            "matrix is $(size(A,1))×$(size(A,2)), but diagonal is $(size(D,1))×$(size(D,2))"
        ),
    )
    return A * inv(D)
end

function LinearAlgebra.ldiv!(D::DiagonalNDArray, B::NDArray)
    return copyto!(B, D \ B)
end

function LinearAlgebra.rdiv!(A::NDArray, D::DiagonalNDArray)
    return copyto!(A, A / D)
end

function Base.inv(D::DiagonalNDArray{T}) where {T}
    # Match Base LinearAlgebra: throw SingularException at the first zero diagonal entry.
    d = Array(_diag_vec(D))
    for i in eachindex(d)
        if iszero(d[i])
            throw(SingularException(i))
        end
    end
    return Diagonal(inv.(_diag_vec(D)))
end

# Match Base: det(D::Diagonal) = prod(D.diag). Unwrap the 0D NDArray reduction
# so LinearAlgebra.det returns a Julia scalar, not an NDArray.
function LinearAlgebra.det(D::DiagonalNDArray)
    return unwrap(prod(_diag_vec(D)))
end

# Base `tr(D::Diagonal) = sum(tr, D.diag)` mapreduces and scalar-indexes NDArray.
LinearAlgebra.tr(D::DiagonalNDArray{<:Number}) = unwrap(sum(_diag_vec(D)))

# Base `_sum(::Diagonal, ::Colon) = sum(D.diag)` leaves a 0D NDArray; unwrap to a scalar.
Base.sum(D::DiagonalNDArray) = unwrap(sum(_diag_vec(D)))

# Generic `prod(::AbstractMatrix)` walks every entry (scalar-indexing). For n>1 a
# Diagonal has off-diagonal zeros, so the product is zero — match Base.
function Base.prod(D::DiagonalNDArray{T}) where {T<:Number}
    n = size(D, 1)
    n == 0 && return one(T)
    n == 1 && return unwrap(prod(_diag_vec(D)))
    return zero(T)
end

function Base.maximum(D::DiagonalNDArray{T}) where {T<:Number}
    maxdiag = unwrap(maximum(_diag_vec(D)))
    size(D, 1) > 1 && return max(zero(T), maxdiag)
    return maxdiag
end

function Base.minimum(D::DiagonalNDArray{T}) where {T<:Number}
    mindiag = unwrap(minimum(_diag_vec(D)))
    size(D, 1) > 1 && return min(zero(T), mindiag)
    return mindiag
end

Base.iszero(D::DiagonalNDArray) = unwrap(all(_diag_vec(D) .== zero(eltype(D))))
Base.isone(D::DiagonalNDArray) = unwrap(all(_diag_vec(D) .== one(eltype(D))))

# istriu/istril with |k|>0 consult `iszero(D.diag)` via scalar iteration in Base.
function LinearAlgebra.istriu(D::DiagonalNDArray, k::Integer=0)
    return k <= 0 || iszero(D)
end
function LinearAlgebra.istril(D::DiagonalNDArray, k::Integer=0)
    return k >= 0 || iszero(D)
end

# Real Diagonal is always Hermitian/symmetric in Base; Complex Hermitian needs isreal(diag).
LinearAlgebra.ishermitian(D::DiagonalNDArray{<:Real}) = true
function LinearAlgebra.ishermitian(D::DiagonalNDArray{<:Complex})
    return unwrap(all(imag(_diag_vec(D)) .== zero(real(eltype(D)))))
end
LinearAlgebra.issymmetric(D::DiagonalNDArray{<:Number}) = true

# Base `isposdef(D) = all(isposdef, D.diag)` scalar-iterates.
function LinearAlgebra.isposdef(D::DiagonalNDArray{T}) where {T<:Real}
    isempty(D) && return true
    return unwrap(all(_diag_vec(D) .> zero(T)))
end
function LinearAlgebra.isposdef(D::DiagonalNDArray{T}) where {T<:Complex}
    # isposdef(z) = isreal(z) && real(z) > 0 — keep on-device, no host densify.
    d = _diag_vec(D)
    return unwrap(all((imag(d) .== zero(real(T))) .& (real(d) .> zero(real(T)))))
end

###### Eigen / related ######

# Base: eigvals(D::Diagonal{<:Number}) = copy(D.diag). Keep NDArray (package style).
function LinearAlgebra.eigvals(D::DiagonalNDArray{<:Number}; permute::Bool=true, scale::Bool=true)
    copy(_diag_vec(D))
end

# Unsorted eigen: values are a copy of the diagonal (NDArray); vectors are NDArray I.
# Keyword `sortby` is not accepted on this override.
function LinearAlgebra.eigen(
    D::DiagonalNDArray;
    permute::Bool=true,
    scale::Bool=true,
)
    Td = Base.promote_op(/, eltype(D), eltype(D))
    return Eigen(copy(_diag_vec(D)), _eye(Td, size(D, 1)))
end

function LinearAlgebra.eigvecs(
    D::DiagonalNDArray;
    permute::Bool=true,
    scale::Bool=true,
)
    return eigen(D; permute=permute, scale=scale).vectors
end

# Real logdet is sum(log.(diag)) on-device. Complex logdet / other Base LinearAlgebra
# ops without overrides fall through and may scalar-index `.diag`.
LinearAlgebra.logdet(D::DiagonalNDArray{<:Real}) = unwrap(sum(log.(_diag_vec(D))))

# Operator / entrywise norms from the diagonal only (no host densify).
function LinearAlgebra.opnorm(D::DiagonalNDArray, p::Real=2)
    if !(p == 1 || p == 2 || p == Inf)
        throw(ArgumentError(lazy"invalid p-norm p=$p. Valid: 1, 2, Inf"))
    end
    isempty(D) && return float(real(zero(eltype(D))))
    return unwrap(maximum(abs.(_diag_vec(D))))
end

function LinearAlgebra.norm(D::DiagonalNDArray, p::Real=2)
    # Off-diagonals are zero, so the matrix vec-norm equals the diag vec-norm.
    d = abs.(_diag_vec(D))
    if p == 2
        return sqrt(unwrap(sum(d .^ 2)))
    elseif p == 1
        return unwrap(sum(d))
    elseif p == Inf
        return isempty(D) ? float(real(zero(eltype(D)))) : unwrap(maximum(d))
    elseif p == -Inf
        return isempty(D) ? float(real(zero(eltype(D)))) : unwrap(minimum(d))
    else
        return unwrap(sum(d .^ p))^(one(p) / p)
    end
end

function LinearAlgebra.cond(D::DiagonalNDArray, p::Real=2)
    if !(p == 1 || p == 2 || p == Inf)
        throw(ArgumentError(lazy"invalid p-norm p=$p. Valid: 1, 2, Inf"))
    end
    isempty(D) && return float(one(real(eltype(D))))
    dabs = abs.(_diag_vec(D))
    return unwrap(maximum(dabs)) / unwrap(minimum(dabs))
end

function Base.:+(A::NDArray{T,2}, D::DiagonalNDArray) where {T}
    size(A, 1) == size(A, 2) == size(D, 1) || throw(
        DimensionMismatch(
            "matrix is $(size(A,1))×$(size(A,2)), but diagonal is $(size(D,1))×$(size(D,2))"
        ),
    )
    return A + (_row_scale(_diag_vec(D)) .* _eye(eltype(D), size(D, 1)))
end
Base.:+(D::DiagonalNDArray, A::NDArray{<:Any,2}) = A + D

Base.:-(A::NDArray{<:Any,2}, D::DiagonalNDArray) = A + (-D)
Base.:-(D::DiagonalNDArray, A::NDArray{<:Any,2}) = D + (-A)

###### UniformScaling (LinearAlgebra.I) ######

@inline function _uniformscaling_eye(::Type{R}, n::Integer, λ) where {R}
    E = _eye(R, Int(n))
    return isone(λ) ? E : nda_multiply_scalar(E, R(λ))
end

function NDArray{T}(J::LinearAlgebra.UniformScaling, dims::Dims{2}) where {T}
    A = zeros(T, dims)
    copyto!(A, J)
    return A
end
function NDArray{T}(J::LinearAlgebra.UniformScaling, m::Integer, n::Integer) where {T}
    return NDArray{T}(J, Dims((Int(m), Int(n))))
end
NDArray(J::LinearAlgebra.UniformScaling{T}, dims::Dims{2}) where {T} = NDArray{T}(J, dims)
function NDArray(J::LinearAlgebra.UniformScaling{T}, m::Integer, n::Integer) where {T}
    return NDArray{T}(J, Dims((Int(m), Int(n))))
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
    return _eye(T, size(A, 1))
end
function Base.oneunit(A::NDArray{T,2}) where {T}
    LinearAlgebra.checksquare(A)
    return _eye(T, size(A, 1))
end

###### Diagonal ↔ UniformScaling ######

# Keep Diagonal structure: D + λI == Diagonal(d .+ λ), not a dense matrix.
function Base.:+(D::DiagonalNDArray{T}, J::LinearAlgebra.UniformScaling) where {T}
    R = Base.promote_op(+, T, typeof(J.λ))
    return Diagonal(_diag_vec(D) .+ convert(R, J.λ))
end
Base.:+(J::LinearAlgebra.UniformScaling, D::DiagonalNDArray) = D + J

Base.:-(D::DiagonalNDArray, J::LinearAlgebra.UniformScaling) = D + (-J)
function Base.:-(J::LinearAlgebra.UniformScaling, D::DiagonalNDArray{T}) where {T}
    R = Base.promote_op(-, typeof(J.λ), T)
    return Diagonal(convert(R, J.λ) .- _diag_vec(D))
end

function Base.:*(D::DiagonalNDArray{T}, J::LinearAlgebra.UniformScaling) where {T}
    return Diagonal(_mul_scalar(T, J.λ, _diag_vec(D)))
end
Base.:*(J::LinearAlgebra.UniformScaling, D::DiagonalNDArray) = D * J

function Base.copyto!(D::DiagonalNDArray{T}, J::LinearAlgebra.UniformScaling) where {T}
    fill!(_diag_vec(D), convert(T, J.λ))
    return D
end

###### Diagonal broadcast ######

# LinearAlgebra's StructuredMatrixStyle{Diagonal} path (structuredbroadcast.jl)
# fills `dest.diag[i]` via `Broadcast._broadcast_getindex(bc, (i,i))`, which
# scalar-indexes the NDArray. CUDA/GPUArrays hit the same Base path and do not
# special-case Diagonal broadcast either.
#
# For structure-preserving / zero-preserving broadcasts we lower to a 1D
# broadcast on `.diag` (NDArrayStyle). Fusion then applies to that vector
# broadcast as usual; there is no separate Diagonal-matrix fusion.
#
# Densifying broadcasts (e.g. `D .+ 1`, `D .+= 1`) cannot be lowered that way.
# Base densifies out-of-place and rejects in-place off-diagonal writes; both
# paths would scalar-index NDArray-backed Diagonal first. Reject early with an
# ArgumentError (matching Base's off-diagonal message when we can evaluate it
# without touching `.diag`).

@inline _has_diagonal_ndarray(@nospecialize(_)) = false
@inline _has_diagonal_ndarray(::DiagonalNDArray) = true
@inline function _has_diagonal_ndarray(bc::Broadcast.Broadcasted)
    return _has_diagonal_ndarray_args(bc.args)
end
@inline _has_diagonal_ndarray_args(::Tuple{}) = false
@inline function _has_diagonal_ndarray_args(args::Tuple)
    return _has_diagonal_ndarray(getfield(args, 1)) ||
           _has_diagonal_ndarray_args(Base.tail(args))
end

# Dense NDArray leaves (not wrapped in Diagonal). Used to avoid scalar-indexing
# when building the densifying off-diagonal ArgumentError.
@inline _has_plain_ndarray(@nospecialize(_)) = false
@inline _has_plain_ndarray(::NDArray) = true
@inline function _has_plain_ndarray(bc::Broadcast.Broadcasted)
    return _has_plain_ndarray_args(bc.args)
end
@inline _has_plain_ndarray_args(::Tuple{}) = false
@inline function _has_plain_ndarray_args(args::Tuple)
    return _has_plain_ndarray(getfield(args, 1)) ||
           _has_plain_ndarray_args(Base.tail(args))
end

@inline _first_diagonal_ndarray_diag(D::DiagonalNDArray) = D.diag
@inline function _first_diagonal_ndarray_diag(bc::Broadcast.Broadcasted)
    return _first_diagonal_ndarray_diag_args(bc.args)
end
@inline function _first_diagonal_ndarray_diag_args(args::Tuple)
    a = getfield(args, 1)
    return if _has_diagonal_ndarray(a)
        _first_diagonal_ndarray_diag(a)
    else
        _first_diagonal_ndarray_diag_args(Base.tail(args))
    end
end

# Replace Diagonal leaves with their `.diag` vectors; keep scalars / Refs / etc.
@inline _diag_bc_arg(D::Diagonal) = D.diag
@inline function _diag_bc_arg(bc::Broadcast.Broadcasted)
    return Broadcast.broadcasted(bc.f, map(_diag_bc_arg, bc.args)...)
end
@inline _diag_bc_arg(x) = x

@inline function _diagonal_broadcast_preserves_structure(bc::Broadcast.Broadcasted)
    # `+` / `-` on Diagonal+Diagonal are zero-preserving (Base's fzeropreserving),
    # not isstructurepreserving. Prefer either so D.+D lowers to `.diag` broadcast.
    return LinearAlgebra.isstructurepreserving(bc) || LinearAlgebra.fzeropreserving(bc)
end

# Off-diagonal Diagonal getindex uses `diagzero` (no `.diag` read), so evaluating
# an off-diagonal broadcast index is safe without `@allowscalar` when every
# non-Diagonal leaf is a scalar. Dense NDArray leaves must not be indexed.
function _throw_densifying_diagonal_broadcast(bc::Broadcast.Broadcasted)
    axs = axes(bc)
    if length(axs) >= 2 && length(axs[1]) >= 2 && length(axs[2]) >= 2
        if !_has_plain_ndarray(bc)
            v = @inbounds Broadcast._broadcast_getindex(bc, CartesianIndex(2, 1))
            throw(
                ArgumentError(
                    "cannot set off-diagonal entry (2, 1) to a nonzero value ($v)"
                ),
            )
        end
        throw(
            ArgumentError(
                "cannot set off-diagonal entry (2, 1) to a nonzero value; " *
                "broadcast over Diagonal with NDArray diagonal would densify",
            ),
        )
    end
    throw(
        ArgumentError(
            "broadcast over Diagonal with NDArray diagonal is not structure-preserving " *
            "and would densify; only structure- or zero-preserving broadcasts are supported",
        ),
    )
end

function Base.similar(
    bc::Broadcast.Broadcasted{LinearAlgebra.StructuredMatrixStyle{Diagonal}},
    ::Type{ElType},
) where {ElType}
    inds = axes(bc)
    n = length(inds[1])
    if _has_diagonal_ndarray(bc)
        # n == 1 has no off-diagonals; keep Diagonal even when not zero-preserving.
        if _diagonal_broadcast_preserves_structure(bc) || n <= 1
            d = _first_diagonal_ndarray_diag(bc)
            return Diagonal(similar(d, ElType, (n,)))
        end
        return _throw_densifying_diagonal_broadcast(bc)
    elseif _diagonal_broadcast_preserves_structure(bc)
        return LinearAlgebra.structured_broadcast_alloc(bc, Diagonal, ElType, n)
    else
        return similar(
            convert(Broadcast.Broadcasted{Broadcast.DefaultArrayStyle{ndims(bc)}}, bc),
            ElType,
        )
    end
end

@inline function _copyto_diagonal_ndarray!(
    dest::DiagonalNDArray, bc::Broadcast.Broadcasted
)
    axes(bc) == axes(dest) || Broadcast.throwdm(axes(bc), axes(dest))
    # Lower to NDArrayStyle vector broadcast so `_copyto!` / fusion apply.
    copyto!(_diag_vec(dest), Broadcast.instantiate(_diag_bc_arg(bc)))
    return dest
end

function Base.copyto!(
    dest::DiagonalNDArray,
    bc::Broadcast.Broadcasted{<:LinearAlgebra.StructuredMatrixStyle},
)
    if !LinearAlgebra.isvalidstructbc(dest, bc)
        # 1×1: Base's generic path only writes the diagonal; no off-diagonals to reject.
        size(dest, 1) <= 1 || return _throw_densifying_diagonal_broadcast(bc)
    end
    return _copyto_diagonal_ndarray!(dest, bc)
end

# Safety net if a densifying Structured broadcast is converted to Nothing
# (Base's `isvalidstructbc` fallback) before reaching the method above.
function Base.copyto!(dest::DiagonalNDArray, bc::Broadcast.Broadcasted{Nothing})
    if !_diagonal_broadcast_preserves_structure(bc)
        size(dest, 1) <= 1 || return _throw_densifying_diagonal_broadcast(bc)
    end
    return _copyto_diagonal_ndarray!(dest, bc)
end
