# Type/dim guards dispatch on one argument at a time, then forward to `_solve`.
"""
    cuNumeric.solve(A, b)

Solve linear system(s) `A * x = b`.

`A` must have shape `(..., m, m)`. `b` must have shape `(..., m)` or `(..., m, n)`.
The result has the same shape as `b`. Batch dimensions are supported; the
implementation always uses the batched Legate `SOLVE` path.

Accepted element types are `Float32`, `Float64`, `ComplexF32`, and `ComplexF64`.
Integer or `Bool` inputs promote to `Float64` only when promotion is allowed.
"""
function solve(a::NDArray{<:_SOLVE_ACCEPTED}, b::NDArray{<:_SOLVE_ACCEPTED})
    A, B = eltype(a), eltype(b)
    O = promote_type(_solve_eltype(A), _solve_eltype(B))
    # int/bool -> float is an implicit promotion, disallowed unless `allowpromotion`
    A <: _SOLVE_PROMOTABLE && assertpromotion(solve, A, O)
    B <: _SOLVE_PROMOTABLE && assertpromotion(solve, B, O)
    return _solve_check_a_dims(unchecked_promote_arr(a, O), unchecked_promote_arr(b, O))
end

function solve(a::NDArray, b::NDArray)
    bad = eltype(a) <: _SOLVE_ACCEPTED ? eltype(b) : eltype(a)
    throw(ArgumentError("array type $bad is unsupported in solve"))
end

function svd(a::NDArray{<:_SVD_ACCEPTED}, full_matrices::Bool=true)
    A = eltype(a)
    O = _svd_eltype(A)
    A <: _SVD_PROMOTABLE && assertpromotion(svd, A, O)
    return _svd_check_dims(unchecked_promote_arr(a, O), full_matrices)
end

function svd(a::NDArray, full_matrices::Bool=true)
    throw(ArgumentError("array type $(eltype(a)) is unsupported in svd"))
end

function qr(a::NDArray{<:_QR_ACCEPTED})
    A = eltype(a)
    O = _qr_eltype(A)
    A <: _QR_PROMOTABLE && assertpromotion(qr, A, O)
    return _qr_check_dims(unchecked_promote_arr(a, O))
end

function qr(a::NDArray)
    throw(ArgumentError("array type $(eltype(a)) is unsupported in qr"))
end
