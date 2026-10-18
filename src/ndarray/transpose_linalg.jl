# Base already constructs lazy Transpose/Adjoint wrappers for NDArrays. Keep
# their parents on the backend by expressing transposition as contraction modes.
const _NDMatrix = NDArray{<:SUPPORTED_ARRAY_TYPES,2}
const _NDTransposeMatrix = Transpose{<:Any,<:_NDMatrix}
const _NDAdjointMatrix = Adjoint{<:Any,<:_NDMatrix}
const _NDWrappedMatrix = Union{_NDTransposeMatrix,_NDAdjointMatrix}

_matmul_parent(A::NDArray) = A
_matmul_parent(A::_NDWrappedMatrix) = parent(A)
_matmul_modes(::NDArray, modes) = modes
_matmul_modes(::_NDWrappedMatrix, modes) = reverse(modes)
_matmul_operand(A) = _matmul_parent(A)
_matmul_operand(A::_NDAdjointMatrix) = conj(parent(A))

function _wrapped_matmul!(C::NDArray, A, B, α, β)
    size(A, 2) == size(B, 1) ||
        throw(DimensionMismatch("matrix multiplication dimensions do not match"))
    expected = ndims(B) == 1 ? (size(A, 1),) : (size(A, 1), size(B, 2))
    size(C) == expected ||
        throw(DimensionMismatch("mul! output is $(size(C)), expected $expected"))
    Ap, Bp = _matmul_parent(A), _matmul_parent(B)
    # Check original stores before promotion or conjugation can hide aliasing.
    (nda_overlaps(C, Ap) || nda_overlaps(C, Bp)) &&
        throw(ArgumentError("mul! output must not alias either input"))
    cm, am, bm = ndims(B) == 1 ? ("i", "ij", "j") : ("ij", "ik", "kj")
    am, bm = _matmul_modes(A, am), _matmul_modes(B, bm)
    # The existing zero/empty path must not read either input.
    skip = _host_iszero(α) || isempty(Ap) || isempty(Bp) || isempty(C)
    Ac = skip ? Ap : _matmul_operand(A)
    try
        Bc = skip ? Bp : _matmul_operand(B)
        try
            return _linalg_mul!(C, cm, Ac, am, Bc, bm, α, β)
        finally
            Bc !== Bp && destroy!(Bc)
        end
    finally
        Ac !== Ap && destroy!(Ac)
    end
end

function _wrapped_matmul(A, B)
    T = _matmul_eltype(promote_type(eltype(A), eltype(B)))
    size(A, 2) == size(B, 1) ||
        throw(DimensionMismatch("matrix multiplication dimensions do not match"))
    dims = ndims(B) == 1 ? (size(A, 1),) : (size(A, 1), size(B, 2))
    C = _undef_array(T, dims)
    try
        return _wrapped_matmul!(C, A, B, true, false)
    catch
        destroy!(C)
        rethrow()
    end
end

# Separate wrapper signatures also resolve LinearAlgebra's wrapper-specific
# methods. Leave the existing unwrapped NDArray multiplication paths unchanged.
for AT in (_NDMatrix, _NDTransposeMatrix, _NDAdjointMatrix),
    BT in (_NDMatrix, _NDTransposeMatrix, _NDAdjointMatrix, NDArray{<:SUPPORTED_ARRAY_TYPES,1})
    AT === _NDMatrix && (BT === _NDMatrix || BT <: AbstractVector) && continue
    CT = BT <: AbstractVector ? NDArray{<:SUPPORTED_ARRAY_TYPES,1} : _NDMatrix
    @eval begin
        Base.:*(A::$AT, B::$BT) = _wrapped_matmul(A, B)
        LinearAlgebra.mul!(C::$CT, A::$AT, B::$BT) =
            _wrapped_matmul!(C, A, B, true, false)
        LinearAlgebra.mul!(C::$CT, A::$AT, B::$BT, α::Union{Number,DeviceScalar}, β::Union{Number,DeviceScalar}) =
            _wrapped_matmul!(C, A, B, α, β)
        LinearAlgebra.mul!(C::$CT, A::$AT, B::$BT, α::Number, β::Number) =
            _wrapped_matmul!(C, A, B, α, β)
    end
end
