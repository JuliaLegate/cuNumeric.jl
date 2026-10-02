# Batched linear algebra: operations over a stack of matrices, dispatching on 3D
# arrays. These have no Base.LinearAlgebra counterpart, so they keep the
# `batched_` prefix and return plain tuples rather than `Factorization` objects.
#
# All of them take exactly one batch dimension, i.e. shape (b, m, m). See
# `MAX_BATCHED_DIM` for why.

"""
    cuNumeric.batched_solve(A, B)

Solve a stack of linear systems `A * X = B`.

`A` must have shape `(b, m, m)` and `B` shape `(b, m, n)`. The result has the
same shape as `B`.

Accepted element types are `Float32`, `Float64`, `ComplexF32`, and `ComplexF64`.
Integer and `Bool` inputs are converted to `Float64`. As everywhere else in
the package, that conversion needs `@allowpromotion` only when it widens the
element type, so `Int64` and `UInt64` pass through silently.

For a single system use [`cuNumeric.solve`](@ref).
"""
function batched_solve(
    a::NDArray{A}, b::NDArray{B}
) where {A<:_SOLVE_ACCEPTED,B<:_SOLVE_ACCEPTED}
    O = promote_type(_solve_eltype(A), _solve_eltype(B))
    return _solve_check_a_dims_batched(
        checked_promote_arr(batched_solve, a, O), checked_promote_arr(batched_solve, b, O)
    )
end

function batched_solve(a::NDArray, b::NDArray)
    bad = eltype(a) <: _SOLVE_ACCEPTED ? eltype(b) : eltype(a)
    return throw(ArgumentError("array type $bad is unsupported in batched_solve"))
end

"""
    cuNumeric.batched_cholesky(A)

Cholesky factor of every matrix in the stack `A`, returned as a single array of
the same shape. Each `A[i, :, :]` is factored independently into a lower
triangular `L` with `A[i, :, :] ≈ L * L'`; the upper triangle is zeroed.

`A` must have shape `(b, m, m)`. As in `LinearAlgebra.cholesky`, only the lower
triangle is read and Hermitian-ness is not checked.

Accepted element types are `Float32`, `Float64`, `ComplexF32`, and `ComplexF64`.
Integer and `Bool` inputs are converted to `Float64`. As everywhere else in
the package, that conversion needs `@allowpromotion` only when it widens the
element type, so `Int64` and `UInt64` pass through silently.
"""
function batched_cholesky(a::NDArray{T,N}) where {T<:_CHOLESKY_ACCEPTED,N}
    _assert_batched_dims(:batched_cholesky, N)
    return _cholesky(checked_promote_arr(batched_cholesky, a, _cholesky_eltype(T)))
end

function batched_cholesky(a::NDArray)
    return throw(ArgumentError("array type $(eltype(a)) is unsupported in batched_cholesky"))
end

"""
    cuNumeric.batched_eigen(A) -> (values, vectors)

Eigenvalues and right eigenvectors of every matrix in the stack `A`.

`A` must have shape `(b, m, m)`. `values` has shape `(b, m)` and `vectors` has
the shape of `A`, with `vectors[i, :, j]` the eigenvector for `values[i, j]`.

Both results are always complex, even for real input. See `LinearAlgebra.eigen`.
"""
function batched_eigen(a::NDArray{T,N}) where {T<:_EIG_ACCEPTED,N}
    _assert_batched_dims(:batched_eigen, N)
    return _eig(checked_promote_arr(batched_eigen, a, _eig_eltype(T)))
end

function batched_eigen(a::NDArray)
    return throw(ArgumentError("array type $(eltype(a)) is unsupported in batched_eigen"))
end

"""
    cuNumeric.batched_eigvals(A)

Eigenvalues of every matrix in the stack `A`, as a complex array of shape
`(b, m)`. See [`cuNumeric.batched_eigen`](@ref).
"""
function batched_eigvals(a::NDArray{T,N}) where {T<:_EIG_ACCEPTED,N}
    _assert_batched_dims(:batched_eigvals, N)
    return _eigvals(checked_promote_arr(batched_eigvals, a, _eig_eltype(T)))
end

function batched_eigvals(a::NDArray)
    return throw(ArgumentError("array type $(eltype(a)) is unsupported in batched_eigvals"))
end
