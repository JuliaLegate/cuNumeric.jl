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
Integer or `Bool` inputs promote to `Float64` only when promotion is allowed.

For a single system use [`cuNumeric.solve`](@ref).
"""
function batched_solve(a::NDArray{<:_SOLVE_ACCEPTED}, b::NDArray{<:_SOLVE_ACCEPTED})
    A, B = eltype(a), eltype(b)
    O = promote_type(_solve_eltype(A), _solve_eltype(B))
    A <: _SOLVE_PROMOTABLE && assertpromotion(batched_solve, A, O)
    B <: _SOLVE_PROMOTABLE && assertpromotion(batched_solve, B, O)
    return _solve_check_a_dims_batched(
        unchecked_promote_arr(a, O), unchecked_promote_arr(b, O)
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
Integer or `Bool` inputs promote to `Float64` only when promotion is allowed.
"""
function batched_cholesky(a::NDArray{<:_CHOLESKY_ACCEPTED,N}) where {N}
    _assert_batched_dims(:batched_cholesky, N)
    A = eltype(a)
    O = _cholesky_eltype(A)
    A <: _CHOLESKY_PROMOTABLE && assertpromotion(batched_cholesky, A, O)
    return _cholesky(unchecked_promote_arr(a, O))
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
function batched_eigen(a::NDArray{<:_EIG_ACCEPTED,N}) where {N}
    _assert_batched_dims(:batched_eigen, N)
    return _eig(_promote_for_eig(a))
end

function batched_eigen(a::NDArray)
    return throw(ArgumentError("array type $(eltype(a)) is unsupported in batched_eigen"))
end

"""
    cuNumeric.batched_eigvals(A)

Eigenvalues of every matrix in the stack `A`, as a complex array of shape
`(b, m)`. See [`cuNumeric.batched_eigen`](@ref).
"""
function batched_eigvals(a::NDArray{<:_EIG_ACCEPTED,N}) where {N}
    _assert_batched_dims(:batched_eigvals, N)
    return _eigvals(_promote_for_eig(a))
end

function batched_eigvals(a::NDArray)
    return throw(ArgumentError("array type $(eltype(a)) is unsupported in batched_eigvals"))
end
