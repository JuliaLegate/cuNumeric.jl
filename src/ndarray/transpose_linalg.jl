const _NDVectorWrapper = Union{
    LinearAlgebra.Transpose{<:Any,<:NDArray{<:SUPPORTED_ARRAY_TYPES,1}},
    LinearAlgebra.Adjoint{<:Any,<:NDArray{<:SUPPORTED_ARRAY_TYPES,1}},
}
const _NDWrapper = Union{_NDTransposeMatrix,_NDVectorWrapper}
const _NDMatrixOperand = Union{NDArray{<:SUPPORTED_ARRAY_TYPES,2},_NDTransposeMatrix}

_wrapper_parent(A::NDArray) = A
_wrapper_parent(A::_NDWrapper) = parent(A)

# Every returned handle is owned by the caller. A transpose/reshape store view
# needs no data copy; complex adjoints materialize conjugation on the backend.
function _wrapper_array(A::_NDWrapper; independent=false, conjugate=true)
    P = parent(A)
    view = ndims(P) == 1 ? nda_reshape_array(P, (1, length(P))) : nda_transpose(P)
    conjugate = conjugate && A isa LinearAlgebra.Adjoint && eltype(P) <: Complex
    !(independent || conjugate) && return view
    try
        return conjugate ? conj.(view) : copy(view)
    finally
        destroy!(view)
    end
end

NDArray(A::_NDWrapper) = _wrapper_array(A; independent=true)
for W in (:Transpose, :Adjoint)
    @eval Base.copy(A::LinearAlgebra.$W{<:Any,<:NDArray{<:Any,2}}) = NDArray(A)
    @eval Base.copy(A::LinearAlgebra.$W{<:Any,<:NDArray{<:Any,1}}) = LinearAlgebra.$W(copy(parent(A)))
    for N in (1, 2)
        @eval Base.BroadcastStyle(::Type{<:LinearAlgebra.$W{<:Any,<:NDArray{<:Any,$N}}}) = NDArrayStyle{2}()
    end
end
__materialize(A::_NDWrapper) = _wrapper_array(A)

_host_wrapper(A::LinearAlgebra.Transpose) = Base.transpose(Array(parent(A)))
_host_wrapper(A::LinearAlgebra.Adjoint) = adjoint(Array(parent(A)))
Base.Array(A::_NDWrapper) = Array(_host_wrapper(A))
Base.Array{T}(A::_NDWrapper) where {T} = Array{T}(_host_wrapper(A))
Base.Array{T,2}(A::_NDWrapper) where {T} = Array{T,2}(_host_wrapper(A))
Base.collect(A::_NDWrapper) = Array(A)
Base.collect(::Type{T}, A::_NDWrapper) where {T} = Array{T}(A)

function Base.copyto!(dest::NDArray{T,2}, src::_NDWrapper) where {T}
    size(dest) == size(src) || throw(DimensionMismatch("transpose/adjoint copy dimensions do not match"))
    T === eltype(src) || throw(ArgumentError("copyto! requires matching element types; convert the source explicitly"))
    tmp = _wrapper_array(src; independent=nda_overlaps(dest, parent(src)))
    try
        return copyto!(dest, tmp)
    finally
        destroy!(tmp)
    end
end
for N in (1, 2)
    @eval LinearAlgebra.transpose!(dest::NDArray{<:Any,2}, src::NDArray{<:Any,$N}) = copyto!(dest, LinearAlgebra.Transpose(src))
    @eval LinearAlgebra.adjoint!(dest::NDArray{<:Any,2}, src::NDArray{<:Any,$N}) = copyto!(dest, LinearAlgebra.Adjoint(src))
end

function Base.copyto!(dest::_NDWrapper, src::NDArray{<:Any,2})
    size(dest) == size(src) || throw(DimensionMismatch("transpose/adjoint copy dimensions do not match"))
    P = parent(dest)
    storage = ndims(P) == 1 ? nda_reshape_array(P, (length(P), 1)) : P
    wrap = dest isa LinearAlgebra.Adjoint ? adjoint : Base.transpose
    try
        copyto!(storage, wrap(src))
        return dest
    finally
        storage !== P && destroy!(storage)
    end
end

function Base.fill!(dest::_NDWrapper, value::Number)
    fill!(parent(dest), dest isa LinearAlgebra.Adjoint ? conj(value) : value)
    return dest
end

function Base.copyto!(dest::_NDWrapper, src::_NDWrapper)
    tmp = _wrapper_array(src)
    try
        return copyto!(dest, tmp)
    finally
        destroy!(tmp)
    end
end

function _materialize_with_policy!(policy::BroadcastOffsetPolicy, dest::_NDWrapper, bc::Broadcasted)
    tmp = _undef_array(eltype(dest), size(dest))
    try
        _materialize_with_policy!(policy, tmp, bc)
        return copyto!(dest, tmp)
    finally
        destroy!(tmp)
    end
end
Base.copyto!(dest::_NDWrapper, bc::Broadcasted{Nothing}) =
    _materialize_with_policy!(FixedOffsets{Int64}(), dest, bc)
for Style in (NDArrayStyle, DefaultArrayStyle{0})
    @eval Base.copyto!(dest::_NDWrapper, bc::Broadcasted{<:$Style}) =
        _materialize_with_policy!(FixedOffsets{Int64}(), dest, bc)
end

_matrix_operand(A::NDArray{<:Any,2}; conjugate=true) = A
_matrix_operand(A::NDArray{<:Any,1}; conjugate=true) = nda_reshape_array(A, (length(A), 1))
_matrix_operand(A::_NDWrapper; conjugate=true) = _wrapper_array(A; conjugate)

function _wrapped_mul!(C::NDArray{<:Any,2}, A, B, α, β)
    size(A, 2) == size(B, 1) && size(C) == (size(A, 1), size(B, 2)) ||
        throw(DimensionMismatch("matrix multiplication dimensions do not match"))
    (nda_overlaps(C, _wrapper_parent(A)) || nda_overlaps(C, _wrapper_parent(B))) &&
        throw(ArgumentError("mul! output must not alias either input"))
    conjugate = !(_host_iszero(α) || isempty(A) || isempty(B) || isempty(C))
    a = _matrix_operand(A; conjugate)
    try
        b = _matrix_operand(B; conjugate)
        try
            return _linalg_mul!(C, "ij", a, "ik", b, "kj", α, β)
        finally
            b !== B && destroy!(b)
        end
    finally
        a !== A && destroy!(a)
    end
end

function _wrapped_product(A, B)
    T = _matmul_eltype(promote_type(eltype(A), eltype(B)))
    size(A, 2) == size(B, 1) || throw(DimensionMismatch("matrix multiplication dimensions do not match"))
    C = _undef_array(T, (size(A, 1), size(B, 2)))
    try
        return _wrapped_mul!(C, A, B, true, false)
    catch
        destroy!(C)
        rethrow()
    end
end

for (AT, BT) in ((_NDTransposeMatrix, NDArray{<:SUPPORTED_ARRAY_TYPES,2}),
                 (NDArray{<:SUPPORTED_ARRAY_TYPES,2}, _NDTransposeMatrix),
                 (_NDTransposeMatrix, _NDTransposeMatrix),
                 (NDArray{<:SUPPORTED_ARRAY_TYPES,1}, _NDVectorWrapper))
    @eval begin
        Base.:*(A::$AT, B::$BT) = _wrapped_product(A, B)
        LinearAlgebra.mul!(C::NDArray{<:SUPPORTED_ARRAY_TYPES,2}, A::$AT, B::$BT) = _wrapped_mul!(C, A, B, true, false)
        LinearAlgebra.mul!(C::NDArray{<:SUPPORTED_ARRAY_TYPES,2}, A::$AT, B::$BT, α::Union{Number,DeviceScalar}, β::Union{Number,DeviceScalar}) = _wrapped_mul!(C, A, B, α, β)
        LinearAlgebra.mul!(C::NDArray{<:SUPPORTED_ARRAY_TYPES,2}, A::$AT, B::$BT, α::Number, β::Number) = _wrapped_mul!(C, A, B, α, β)
    end
end

function Base.:*(A::_NDTransposeMatrix, x::NDArray{<:SUPPORTED_ARRAY_TYPES,1})
    T = _matmul_eltype(promote_type(eltype(A), eltype(x)))
    size(A, 2) == length(x) || throw(DimensionMismatch("matrix-vector dimensions do not match"))
    y = _undef_array(T, (size(A, 1),))
    try
        return mul!(y, A, x)
    catch
        destroy!(y)
        rethrow()
    end
end

# Preserve Julia's row-vector result, including mixed transpose/adjoint products.
function _row_product(u::_NDVectorWrapper, A::_NDMatrixOperand)
    a = _matrix_operand(A)
    wrap = u isa LinearAlgebra.Adjoint ? adjoint : Base.transpose
    try
        return wrap(wrap(a) * parent(u))
    finally
        a !== A && destroy!(a)
    end
end
for W in (:Transpose, :Adjoint)
    @eval Base.:*(u::LinearAlgebra.$W{<:Any,<:NDArray{<:SUPPORTED_ARRAY_TYPES,1}}, A::_NDMatrixOperand) = _row_product(u, A)
end

# Adjoint-vector products already delegate to dot. A complex transpose must
# instead use the nonconjugating backend dot, with the same promotion rules.
function Base.:*(u::LinearAlgebra.Transpose{<:Any,<:NDArray{<:SUPPORTED_ARRAY_TYPES,1}}, v::NDArray{<:SUPPORTED_ARRAY_TYPES,1})
    x = parent(u)
    length(x) == length(v) || throw(DimensionMismatch("dot vector lengths do not match"))
    T = _dot_eltype(promote_type(eltype(x), eltype(v)))
    xp = checked_promote_arr(dot, x, T)
    try
        vp = checked_promote_arr(dot, v, T)
        try
            return cnscalar(isempty(x) ? cuNumeric.zeros(T, ()) : nda_dot(xp, vp))
        finally
            vp !== v && destroy!(vp)
        end
    finally
        xp !== x && destroy!(xp)
    end
end
Base.:*(u::LinearAlgebra.Transpose{T,<:NDArray{<:SUPPORTED_ARRAY_TYPES,1}}, v::NDArray{T,1}) where {T<:Union{_LA_INTEGER,SUPPORTED_FLOAT_TYPES}} = dot(parent(u), v)
