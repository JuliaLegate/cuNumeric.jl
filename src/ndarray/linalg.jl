export NDArrayQR

# Type/dim guards dispatch on one argument at a time, then forward to `_solve`.
"""
    cuNumeric.solve(A, b)

Solve the linear system `A * x = b`.

`A` must be a square `(m, m)` matrix. `b` must have shape `(m,)` or `(m, n)`.
The result has the same shape as `b`.

Accepted element types are `Float32`, `Float64`, `ComplexF32`, and `ComplexF64`.
Integer or `Bool` inputs promote to `Float64` only when promotion is allowed.

For a stack of systems of shape `(b, m, m)` use
[`cuNumeric.batched_solve`](@ref).

See also `\\`.
"""
function solve(a::NDArray{<:_SOLVE_ACCEPTED}, b::NDArray{<:_SOLVE_ACCEPTED})
    A, B = eltype(a), eltype(b)
    O = promote_type(_solve_eltype(A), _solve_eltype(B))
    # int/bool -> float is an implicit promotion, disallowed unless `allowpromotion`
    A <: _SOLVE_PROMOTABLE && assertpromotion(solve, A, O)
    B <: _SOLVE_PROMOTABLE && assertpromotion(solve, B, O)
    return _solve_check_a_dims_2d(unchecked_promote_arr(a, O), unchecked_promote_arr(b, O))
end

function solve(a::NDArray, b::NDArray)
    bad = eltype(a) <: _SOLVE_ACCEPTED ? eltype(b) : eltype(a)
    return throw(ArgumentError("array type $bad is unsupported in solve"))
end

"""
    A \\ b

Solve `A * x = b` for a square 2D `NDArray` `A`. Equivalent to
[`cuNumeric.solve`](@ref).
"""
Base.:\(a::NDArray{<:Any,2}, b::NDArray{<:Any,1}) = solve(a, b)
Base.:\(a::NDArray{<:Any,2}, b::NDArray{<:Any,2}) = solve(a, b)

"""
    LinearAlgebra.cholesky(A::NDArray{T,2}) -> Cholesky

Cholesky factorization of the Hermitian positive-definite matrix `A`, returned as
a `LinearAlgebra.Cholesky` object holding the lower factor `L` with `A ≈ L * L'`.

Only the lower triangle of `A` is read and, unlike Base, it is *not* checked for
being Hermitian. A non-positive-definite input raises an `ErrorException` from
the task rather than `LinearAlgebra.PosDefException`, so the `check` keyword is
not supported.

Accepted element types are `Float32`, `Float64`, `ComplexF32`, and `ComplexF64`.
Integer or `Bool` inputs promote to `Float64` only when promotion is allowed.

For a stack of matrices of shape `(b, m, m)` use
[`cuNumeric.batched_cholesky`](@ref).
"""
function LinearAlgebra.cholesky(a::NDArray{<:_CHOLESKY_ACCEPTED,2})
    return LinearAlgebra.Cholesky(_promoted_cholesky(a), 'L', 0)
end

function LinearAlgebra.cholesky(a::NDArray{<:Any,2})
    return throw(ArgumentError("array type $(eltype(a)) is unsupported in cholesky"))
end

function _promoted_cholesky(a::NDArray)
    A = eltype(a)
    O = _cholesky_eltype(A)
    A <: _CHOLESKY_PROMOTABLE && assertpromotion(cholesky, A, O)
    return _cholesky(unchecked_promote_arr(a, O))
end

"""
    LinearAlgebra.eigen(A::NDArray{T,2}) -> Eigen

Eigenvalues and right eigenvectors of the square matrix `A`, returned as a
`LinearAlgebra.Eigen` object. Column `j` of `F.vectors` is the eigenvector for
`F.values[j]`.

Values and vectors are **always complex**, even when `A` is real with real
eigenvalues. This follows the underlying LAPACK `geev` path and differs from
Base, which returns real factors for such input.

Accepted element types are `Float32`, `Float64`, `ComplexF32`, and `ComplexF64`.
Integer or `Bool` inputs promote to `Float64` only when promotion is allowed.

For a stack of matrices of shape `(b, m, m)` use
[`cuNumeric.batched_eigen`](@ref).
"""
function LinearAlgebra.eigen(a::NDArray{<:_EIG_ACCEPTED,2})
    return LinearAlgebra.Eigen(_eig(_promote_for_eig(a))...)
end

function LinearAlgebra.eigen(a::NDArray{<:Any,2})
    return throw(ArgumentError("array type $(eltype(a)) is unsupported in eigen"))
end

"""
    LinearAlgebra.eigvals(A::NDArray{T,2})

Eigenvalues of the square matrix `A` as a complex `NDArray`. See
`LinearAlgebra.eigen` for the supported element types and for why the
result is always complex.
"""
LinearAlgebra.eigvals(a::NDArray{<:_EIG_ACCEPTED,2}) = _eigvals(_promote_for_eig(a))

function LinearAlgebra.eigvals(a::NDArray{<:Any,2})
    return throw(ArgumentError("array type $(eltype(a)) is unsupported in eigvals"))
end

"""
    LinearAlgebra.eigvecs(A::NDArray{T,2})

Right eigenvectors of the square matrix `A`, as columns of a complex `NDArray`.
See `LinearAlgebra.eigen` for the supported element types.
"""
LinearAlgebra.eigvecs(a::NDArray{<:Any,2}) = LinearAlgebra.eigen(a).vectors

function _promote_for_eig(a::NDArray)
    A = eltype(a)
    O = _eig_eltype(A)
    A <: _EIG_PROMOTABLE && assertpromotion(eigen, A, O)
    return unchecked_promote_arr(a, O)
end

"""
    LinearAlgebra.svd(A::NDArray{T,2}; full=false) -> SVD

Singular value decomposition of `A`, returned as a `LinearAlgebra.SVD` object
with `A ≈ F.U * Diagonal(F.S) * F.Vt`.

With `m, n = size(A)` and `k = min(m, n)`, `full=false` gives an `m × k` `F.U`
and a `k × n` `F.Vt`, and `full=true` gives `m × m` and `n × n`. `F.S` has length
`k` and is real-valued for both real and complex input.

Destructuring an `SVD` yields `(U, S, V)` — the adjoint of `F.Vt`, not `F.Vt`
itself. `F.V` is a lazy `Adjoint` wrapper, so operating on it falls back to
scalar indexing until `adjoint(::NDArray)` is implemented; prefer `F.Vt`.

Accepted element types are `Float32`, `Float64`, `ComplexF32`, and `ComplexF64`.
Integer or `Bool` inputs promote to `Float64` only when promotion is allowed.
"""
function LinearAlgebra.svd(a::NDArray{<:_SVD_ACCEPTED,2}; full::Bool=false)
    A = eltype(a)
    O = _svd_eltype(A)
    A <: _SVD_PROMOTABLE && assertpromotion(svd, A, O)
    return LinearAlgebra.SVD(_svd(unchecked_promote_arr(a, O), full)...)
end

function LinearAlgebra.svd(a::NDArray{<:Any,2}; full::Bool=false)
    return throw(ArgumentError("array type $(eltype(a)) is unsupported in svd"))
end

"""
    NDArrayQR <: LinearAlgebra.Factorization

QR factorization of an `NDArray`, holding the explicit factors `Q` and `R` with
`A ≈ Q * R`.

The backend returns materialized factors rather than the packed Householder
representation Base uses, so this is a distinct type from `LinearAlgebra.QR` and
`QRCompactWY`. It destructures as `(Q, R)` and exposes `F.Q` and `F.R`.
"""
struct NDArrayQR{T,M<:NDArray{T,2}} <: LinearAlgebra.Factorization{T}
    Q::M
    R::M
end

Base.iterate(F::NDArrayQR) = (F.Q, Val(:R))
Base.iterate(F::NDArrayQR, ::Val{:R}) = (F.R, Val(:done))
Base.iterate(::NDArrayQR, ::Val{:done}) = nothing
Base.size(F::NDArrayQR) = (size(F.Q, 1), size(F.R, 2))
Base.size(F::NDArrayQR, d::Integer) = d == 1 ? size(F.Q, 1) : size(F.R, d)

function Base.show(io::IO, ::MIME"text/plain", F::NDArrayQR)
    summary(io, F)
    println(io)
    println(io, "Q factor: ", summary(F.Q))
    print(io, "R factor: ", summary(F.R))
    return nothing
end

"""
    LinearAlgebra.qr(A::NDArray{T,2}) -> NDArrayQR

Reduced QR factorization of `A`, with `A ≈ F.Q * F.R`. For an `m × n` input and
`k = min(m, n)`, `F.Q` is `m × k` and `F.R` is `k × n`.

See [`NDArrayQR`](@ref) for why this is not a `LinearAlgebra.QRCompactWY`.

Accepted element types are `Float32`, `Float64`, `ComplexF32`, and `ComplexF64`.
Integer or `Bool` inputs promote to `Float64` only when promotion is allowed.
"""
function LinearAlgebra.qr(a::NDArray{<:_QR_ACCEPTED,2})
    A = eltype(a)
    O = _qr_eltype(A)
    A <: _QR_PROMOTABLE && assertpromotion(qr, A, O)
    return NDArrayQR(_qr(unchecked_promote_arr(a, O))...)
end

function LinearAlgebra.qr(a::NDArray{<:Any,2})
    return throw(ArgumentError("array type $(eltype(a)) is unsupported in qr"))
end
