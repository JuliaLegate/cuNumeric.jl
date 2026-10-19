# Base supplies the lazy wrappers. Lower matrix transposes to backend store
# views only when multiplying, then use the existing NDArray multiplication.
const _NDMatrix = NDArray{<:SUPPORTED_ARRAY_TYPES,2}
const _NDTransposeMatrix = Transpose{<:Any,<:_NDMatrix}
const _NDAdjointMatrix = Adjoint{<:Any,<:_NDMatrix}

_matmul_array(A::NDArray) = A
_matmul_array(A::_NDTransposeMatrix) = nda_transpose(parent(A))

function _transpose_mul(A, B)
    a = _matmul_array(A)
    try
        b = _matmul_array(B)
        try
            return a * b
        finally
            b !== B && destroy!(b)
        end
    finally
        a !== A && destroy!(a)
    end
end

Base.:*(A::_NDTransposeMatrix, B::_NDMatrix) = _transpose_mul(A, B)
Base.:*(A::_NDMatrix, B::_NDTransposeMatrix) = _transpose_mul(A, B)
Base.:*(A::_NDTransposeMatrix, B::_NDTransposeMatrix) = _transpose_mul(A, B)
Base.:*(A::_NDTransposeMatrix, x::NDArray{<:SUPPORTED_ARRAY_TYPES,1}) = _transpose_mul(A, x)

# Reject adjoint matrix operands explicitly rather than scalar-indexing through
# LinearAlgebra's generic methods. Keep separate signatures to avoid ambiguities.
for AT in (_NDMatrix, _NDTransposeMatrix, _NDAdjointMatrix),
    BT in (_NDMatrix, _NDTransposeMatrix, _NDAdjointMatrix, NDArray{<:SUPPORTED_ARRAY_TYPES,1})
    (AT === _NDAdjointMatrix || BT === _NDAdjointMatrix) || continue
    @eval Base.:*(A::$AT, B::$BT) = throw(ArgumentError(
        "NDArray multiplication with Adjoint matrices is not supported. " *
        "Use transpose(A) if a non-conjugating transpose is intended.",
    ))
end
