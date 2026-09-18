#= Copyright 2026 Northwestern University,
 *                   Carnegie Mellon University University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author(s): Ethan Meitz <emeitz@andrew.cmu.edu>
=#

module cuNumericTensorOperationsExt

using cuNumeric
using TensorOperations

const CN = cuNumeric
const TO = TensorOperations
const One = TO.One
const Zero = TO.Zero

struct CuNumericBackend <: TO.AbstractBackend end

TO.select_backend(::typeof(TO.tensoradd!), C::CN.NDArray, A::CN.NDArray) =
    CuNumericBackend()
TO.select_backend(::typeof(TO.tensortrace!), C::CN.NDArray, A::CN.NDArray) =
    CuNumericBackend()
function TO.select_backend(
    ::typeof(TO.tensorcontract!), C::CN.NDArray, A::CN.NDArray, B::CN.NDArray
)
    return CuNumericBackend()
end

function TO.tensoradd_type(
    TC, A::CN.NDArray, pA::TO.Index2Tuple, conjA::Bool
)
    return CN.NDArray{TC,TO.numind(pA)}
end

function TO.tensorcontract_type(
    TC,
    A::CN.NDArray,
    pA::TO.Index2Tuple,
    conjA::Bool,
    B::CN.NDArray,
    pB::TO.Index2Tuple,
    conjB::Bool,
    pAB::TO.Index2Tuple,
)
    Tout = TC <: Union{Integer,Bool} ? Float64 : TC
    return CN.NDArray{Tout,TO.numind(pAB)}
end

function TO.tensoralloc(
    ::Type{<:CN.NDArray{T,N}},
    structure,
    ::Val{istemp}=Val(false),
    allocator=TO.DefaultAllocator(),
) where {T,N,istemp}
    return CN.zeros(T, Tuple(structure))
end

function TO.tensorfree!(C::CN.NDArray, allocator=TO.DefaultAllocator())
    CN.destroy!(C)
    return nothing
end

# ------------------------------------------------------------------------------------------
# Scale factors: One/Zero dispatch, Number wraps to 0D and re-dispatches
# ------------------------------------------------------------------------------------------

function _require_0d_scale(x::CN.NDArray)
    ndims(x) == 0 || throw(
        DimensionMismatch("tensor scale factor must be a Julia Number or a 0-d NDArray")
    )
    return x
end

_as_scale(x::One, ::Type) = x
_as_scale(x::Zero, ::Type) = x
function _as_scale(x::CN.NDArray, ::Type{T}) where {T}
    _require_0d_scale(x)
    return _convert_eltype(x, T)
end
function _as_scale(x::Number, ::Type{T}) where {T}
    isone(x) && return One()
    iszero(x) && return Zero()
    return CN.NDArray(convert(T, x))
end

function _free_scale!(orig, scaled)
    if scaled isa CN.NDArray && !(orig isa CN.NDArray && orig === scaled)
        CN.destroy!(scaled)
    end
    return nothing
end

_accumulate!(C::CN.NDArray, A::CN.NDArray, ::One, ::Zero) = (C.=A; C)
_accumulate!(C::CN.NDArray, A::CN.NDArray, α::CN.NDArray, ::Zero) = (C.=α .* A; C)
_accumulate!(C::CN.NDArray, A::CN.NDArray, ::One, ::One) = (C.=C .+ A; C)
_accumulate!(C::CN.NDArray, A::CN.NDArray, α::CN.NDArray, ::One) = (C.=C .+ α .* A; C)
_accumulate!(C::CN.NDArray, A::CN.NDArray, ::One, β::CN.NDArray) = (C.=β .* C .+ A; C)
function _accumulate!(C::CN.NDArray, A::CN.NDArray, α::CN.NDArray, β::CN.NDArray)
    C .= β .* C .+ α .* A
    return C
end
# α = 0: product does not contribute. Strong-zero for β = 0 does not read C.
_accumulate!(C::CN.NDArray, ::CN.NDArray, ::Zero, ::Zero) = (C.=zero(eltype(C)); C)
_accumulate!(C::CN.NDArray, ::CN.NDArray, ::Zero, ::One) = C
_accumulate!(C::CN.NDArray, ::CN.NDArray, ::Zero, β::CN.NDArray) = (C.=β .* C; C)

function _accumulate!(C::CN.NDArray{T}, A::CN.NDArray, α::Number, β::Number) where {T}
    α′ = _as_scale(α, T)
    β′ = _as_scale(β, T)
    try
        return _accumulate!(C, A, α′, β′)
    finally
        _free_scale!(α, α′)
        _free_scale!(β, β′)
    end
end

function _accumulate!(C::CN.NDArray{T}, A::CN.NDArray, α::CN.NDArray, β::Number) where {T}
    α′ = _as_scale(α, T)
    β′ = _as_scale(β, T)
    try
        return _accumulate!(C, A, α′, β′)
    finally
        _free_scale!(α, α′)
        _free_scale!(β, β′)
    end
end

function _accumulate!(C::CN.NDArray{T}, A::CN.NDArray, α::Number, β::CN.NDArray) where {T}
    α′ = _as_scale(α, T)
    β′ = _as_scale(β, T)
    try
        return _accumulate!(C, A, α′, β′)
    finally
        _free_scale!(α, α′)
        _free_scale!(β, β′)
    end
end

function _convert_eltype(A::CN.NDArray, ::Type{T}) where {T}
    return eltype(A) === T ? A : CN.as_type(A, T)
end

function _ensure_backend(op, backend, C, select_args...)
    if backend isa TO.DefaultBackend
        return TO.select_backend(op, C, select_args...)
    elseif backend isa CuNumericBackend
        return backend
    else
        throw(ArgumentError("Unknown backend $backend for $op and NDArray"))
    end
end

# ------------------------------------------------------------------------------------------
# tensoradd!
# ------------------------------------------------------------------------------------------

function _tensoradd_impl!(C::CN.NDArray, A::CN.NDArray, pA, conjA, α, β)
    TO.argcheck_tensoradd(C, A, pA)
    TO.dimcheck_tensoradd(C, A, pA)
    if C.ptr === A.ptr && !TO.istrivialpermutation(pA)
        throw(ArgumentError("output tensor must not alias a permuted input tensor"))
    end

    opA = conjA ? conj(A) : A
    converted = _convert_eltype(opA, eltype(C))
    permutation = TO.linearize(pA)
    permuted =
        TO.istrivialpermutation(permutation) ? converted :
        permutedims(converted, permutation)

    _accumulate!(C, permuted, α, β)

    permuted !== converted && CN.destroy!(permuted)
    converted !== opA && CN.destroy!(converted)
    opA !== A && CN.destroy!(opA)
    return C
end

function TO.tensoradd!(
    C::CN.NDArray,
    A::CN.NDArray,
    pA::TO.Index2Tuple,
    conjA::Bool,
    α::Number,
    β::Number,
    ::CuNumericBackend,
    allocator=TO.DefaultAllocator(),
)
    return _tensoradd_impl!(C, A, pA, conjA, α, β)
end

function TO.tensoradd!(
    C::CN.NDArray,
    A::CN.NDArray,
    pA::TO.Index2Tuple,
    conjA::Bool,
    α::CN.NDArray,
    β::Union{Number,CN.NDArray},
    ::CuNumericBackend,
    allocator=TO.DefaultAllocator(),
)
    _require_0d_scale(α)
    β isa CN.NDArray && _require_0d_scale(β)
    return _tensoradd_impl!(C, A, pA, conjA, α, β)
end

function TO.tensoradd!(
    C::CN.NDArray,
    A::CN.NDArray,
    pA::TO.Index2Tuple,
    conjA::Bool,
    α::Number,
    β::CN.NDArray,
    ::CuNumericBackend,
    allocator=TO.DefaultAllocator(),
)
    _require_0d_scale(β)
    return _tensoradd_impl!(C, A, pA, conjA, α, β)
end

function TO.tensoradd!(
    C::CN.NDArray, A::CN.NDArray, pA::TO.Index2Tuple, conjA::Bool, α::CN.NDArray, β
)
    return TO.tensoradd!(C, A, pA, conjA, α, β, TO.DefaultBackend())
end
function TO.tensoradd!(
    C::CN.NDArray, A::CN.NDArray, pA::TO.Index2Tuple, conjA::Bool, α::Number, β::CN.NDArray
)
    return TO.tensoradd!(C, A, pA, conjA, α, β, TO.DefaultBackend())
end
function TO.tensoradd!(
    C::CN.NDArray, A::CN.NDArray, pA::TO.Index2Tuple, conjA::Bool, α::CN.NDArray, β, backend
)
    return TO.tensoradd!(C, A, pA, conjA, α, β, backend, TO.DefaultAllocator())
end
function TO.tensoradd!(
    C::CN.NDArray,
    A::CN.NDArray,
    pA::TO.Index2Tuple,
    conjA::Bool,
    α::Number,
    β::CN.NDArray,
    backend,
)
    return TO.tensoradd!(C, A, pA, conjA, α, β, backend, TO.DefaultAllocator())
end
function TO.tensoradd!(
    C::CN.NDArray,
    A::CN.NDArray,
    pA::TO.Index2Tuple,
    conjA::Bool,
    α::CN.NDArray,
    β,
    backend,
    allocator,
)
    b = _ensure_backend(TO.tensoradd!, backend, C, A)
    return TO.tensoradd!(C, A, pA, conjA, α, β, b, allocator)
end
function TO.tensoradd!(
    C::CN.NDArray,
    A::CN.NDArray,
    pA::TO.Index2Tuple,
    conjA::Bool,
    α::Number,
    β::CN.NDArray,
    backend,
    allocator,
)
    b = _ensure_backend(TO.tensoradd!, backend, C, A)
    return TO.tensoradd!(C, A, pA, conjA, α, β, b, allocator)
end

# ------------------------------------------------------------------------------------------
# tensortrace!
# ------------------------------------------------------------------------------------------

function _trace_pairs(A::CN.NDArray, q::TO.Index2Tuple)
    current = A
    current_owned = false
    labels = collect(1:ndims(A))

    for (left, right) in zip(q[1], q[2])
        left_position = findfirst(==(left), labels)::Int
        right_position = findfirst(==(right), labels)::Int
        diagonal = CN.diagonal(current; dims=(left_position, right_position))
        current_owned && CN.destroy!(current)

        reduced = sum(diagonal; dims=ndims(diagonal))
        CN.destroy!(diagonal)
        current = dropdims(reduced; dims=ndims(reduced))
        CN.destroy!(reduced)
        current_owned = true

        deleteat!(labels, max(left_position, right_position))
        deleteat!(labels, min(left_position, right_position))
    end
    return current, current_owned, labels
end

function _tensortrace_impl!(C::CN.NDArray, A::CN.NDArray, p, q, conjA, α, β)
    TO.argcheck_tensortrace(C, A, p, q)
    TO.dimcheck_tensortrace(C, A, p, q)
    C.ptr === A.ptr && throw(ArgumentError("output tensor must not alias input tensor"))

    opA = conjA ? conj(A) : A
    traced, traced_owned, labels = _trace_pairs(opA, q)
    output_labels = TO.linearize(p)
    permutation = Tuple(findfirst(==(label), labels) for label in output_labels)
    ordered = TO.istrivialpermutation(permutation) ? traced :
              permutedims(traced, permutation)
    converted = _convert_eltype(ordered, eltype(C))

    _accumulate!(C, converted, α, β)

    converted !== ordered && CN.destroy!(converted)
    ordered !== traced && CN.destroy!(ordered)
    traced_owned && CN.destroy!(traced)
    opA !== A && CN.destroy!(opA)
    return C
end

function TO.tensortrace!(
    C::CN.NDArray,
    A::CN.NDArray,
    p::TO.Index2Tuple,
    q::TO.Index2Tuple,
    conjA::Bool,
    α::Number,
    β::Number,
    ::CuNumericBackend,
    allocator=TO.DefaultAllocator(),
)
    return _tensortrace_impl!(C, A, p, q, conjA, α, β)
end

function TO.tensortrace!(
    C::CN.NDArray,
    A::CN.NDArray,
    p::TO.Index2Tuple,
    q::TO.Index2Tuple,
    conjA::Bool,
    α::CN.NDArray,
    β::Union{Number,CN.NDArray},
    ::CuNumericBackend,
    allocator=TO.DefaultAllocator(),
)
    _require_0d_scale(α)
    β isa CN.NDArray && _require_0d_scale(β)
    return _tensortrace_impl!(C, A, p, q, conjA, α, β)
end

function TO.tensortrace!(
    C::CN.NDArray,
    A::CN.NDArray,
    p::TO.Index2Tuple,
    q::TO.Index2Tuple,
    conjA::Bool,
    α::Number,
    β::CN.NDArray,
    ::CuNumericBackend,
    allocator=TO.DefaultAllocator(),
)
    _require_0d_scale(β)
    return _tensortrace_impl!(C, A, p, q, conjA, α, β)
end

function TO.tensortrace!(
    C::CN.NDArray,
    A::CN.NDArray,
    p::TO.Index2Tuple,
    q::TO.Index2Tuple,
    conjA::Bool,
    α::CN.NDArray,
    β,
)
    return TO.tensortrace!(C, A, p, q, conjA, α, β, TO.DefaultBackend())
end
function TO.tensortrace!(
    C::CN.NDArray,
    A::CN.NDArray,
    p::TO.Index2Tuple,
    q::TO.Index2Tuple,
    conjA::Bool,
    α::Number,
    β::CN.NDArray,
)
    return TO.tensortrace!(C, A, p, q, conjA, α, β, TO.DefaultBackend())
end
function TO.tensortrace!(
    C::CN.NDArray,
    A::CN.NDArray,
    p::TO.Index2Tuple,
    q::TO.Index2Tuple,
    conjA::Bool,
    α::CN.NDArray,
    β,
    backend,
)
    return TO.tensortrace!(C, A, p, q, conjA, α, β, backend, TO.DefaultAllocator())
end
function TO.tensortrace!(
    C::CN.NDArray,
    A::CN.NDArray,
    p::TO.Index2Tuple,
    q::TO.Index2Tuple,
    conjA::Bool,
    α::Number,
    β::CN.NDArray,
    backend,
)
    return TO.tensortrace!(C, A, p, q, conjA, α, β, backend, TO.DefaultAllocator())
end
function TO.tensortrace!(
    C::CN.NDArray,
    A::CN.NDArray,
    p::TO.Index2Tuple,
    q::TO.Index2Tuple,
    conjA::Bool,
    α::CN.NDArray,
    β,
    backend,
    allocator,
)
    b = _ensure_backend(TO.tensortrace!, backend, C, A)
    return TO.tensortrace!(C, A, p, q, conjA, α, β, b, allocator)
end
function TO.tensortrace!(
    C::CN.NDArray,
    A::CN.NDArray,
    p::TO.Index2Tuple,
    q::TO.Index2Tuple,
    conjA::Bool,
    α::Number,
    β::CN.NDArray,
    backend,
    allocator,
)
    b = _ensure_backend(TO.tensortrace!, backend, C, A)
    return TO.tensortrace!(C, A, p, q, conjA, α, β, b, allocator)
end

# ------------------------------------------------------------------------------------------
# tensorcontract!
# ------------------------------------------------------------------------------------------

function _contract_modes(
    A::CN.NDArray,
    pA::TO.Index2Tuple,
    B::CN.NDArray,
    pB::TO.Index2Tuple,
    pAB::TO.Index2Tuple,
)
    nlabels = ndims(A) + length(pB[2])
    nlabels <= typemax(UInt8) ||
        throw(ArgumentError("tensor contraction requires $nlabels distinct mode labels"))

    Amodes = collect(UInt8, 1:ndims(A))
    Bmodes = Vector{UInt8}(undef, ndims(B))
    for (a, b) in zip(pA[2], pB[1])
        Bmodes[b] = Amodes[a]
    end
    nextlabel = ndims(A)
    for b in pB[2]
        nextlabel += 1
        Bmodes[b] = UInt8(nextlabel)
    end

    free_modes = (map(i -> Amodes[i], pA[1])..., map(i -> Bmodes[i], pB[2])...)
    Cmodes = UInt8[free_modes[i] for i in TO.linearize(pAB)]
    return Cmodes, Amodes, Bmodes
end

function _tensorcontract_impl!(
    C::CN.NDArray, A::CN.NDArray, pA, conjA, B::CN.NDArray, pB, conjB, pAB, α, β
)
    TO.argcheck_tensorcontract(C, A, pA, B, pB, pAB)
    TO.dimcheck_tensorcontract(C, A, pA, B, pB, pAB)

    opA = conjA ? conj(A) : A
    opB = conjB ? conj(B) : B
    convertedA = _convert_eltype(opA, eltype(C))
    convertedB = _convert_eltype(opB, eltype(C))
    Cmodes, Amodes, Bmodes = _contract_modes(convertedA, pA, convertedB, pB, pAB)

    CN.contract!(C, Cmodes, convertedA, Amodes, convertedB, Bmodes; α=α, β=β)

    convertedB !== opB && CN.destroy!(convertedB)
    convertedA !== opA && CN.destroy!(convertedA)
    opB !== B && CN.destroy!(opB)
    opA !== A && CN.destroy!(opA)
    return C
end

function TO.tensorcontract!(
    C::CN.NDArray,
    A::CN.NDArray,
    pA::TO.Index2Tuple,
    conjA::Bool,
    B::CN.NDArray,
    pB::TO.Index2Tuple,
    conjB::Bool,
    pAB::TO.Index2Tuple,
    α::Number,
    β::Number,
    ::CuNumericBackend,
    allocator=TO.DefaultAllocator(),
)
    return _tensorcontract_impl!(C, A, pA, conjA, B, pB, conjB, pAB, α, β)
end

function TO.tensorcontract!(
    C::CN.NDArray,
    A::CN.NDArray,
    pA::TO.Index2Tuple,
    conjA::Bool,
    B::CN.NDArray,
    pB::TO.Index2Tuple,
    conjB::Bool,
    pAB::TO.Index2Tuple,
    α::CN.NDArray,
    β::Union{Number,CN.NDArray},
    ::CuNumericBackend,
    allocator=TO.DefaultAllocator(),
)
    _require_0d_scale(α)
    β isa CN.NDArray && _require_0d_scale(β)
    return _tensorcontract_impl!(C, A, pA, conjA, B, pB, conjB, pAB, α, β)
end

function TO.tensorcontract!(
    C::CN.NDArray,
    A::CN.NDArray,
    pA::TO.Index2Tuple,
    conjA::Bool,
    B::CN.NDArray,
    pB::TO.Index2Tuple,
    conjB::Bool,
    pAB::TO.Index2Tuple,
    α::Number,
    β::CN.NDArray,
    ::CuNumericBackend,
    allocator=TO.DefaultAllocator(),
)
    _require_0d_scale(β)
    return _tensorcontract_impl!(C, A, pA, conjA, B, pB, conjB, pAB, α, β)
end

function TO.tensorcontract!(
    C::CN.NDArray,
    A::CN.NDArray,
    pA::TO.Index2Tuple,
    conjA::Bool,
    B::CN.NDArray,
    pB::TO.Index2Tuple,
    conjB::Bool,
    pAB::TO.Index2Tuple,
    α::CN.NDArray,
    β,
)
    return TO.tensorcontract!(
        C, A, pA, conjA, B, pB, conjB, pAB, α, β, TO.DefaultBackend()
    )
end
function TO.tensorcontract!(
    C::CN.NDArray,
    A::CN.NDArray,
    pA::TO.Index2Tuple,
    conjA::Bool,
    B::CN.NDArray,
    pB::TO.Index2Tuple,
    conjB::Bool,
    pAB::TO.Index2Tuple,
    α::Number,
    β::CN.NDArray,
)
    return TO.tensorcontract!(
        C, A, pA, conjA, B, pB, conjB, pAB, α, β, TO.DefaultBackend()
    )
end
function TO.tensorcontract!(
    C::CN.NDArray,
    A::CN.NDArray,
    pA::TO.Index2Tuple,
    conjA::Bool,
    B::CN.NDArray,
    pB::TO.Index2Tuple,
    conjB::Bool,
    pAB::TO.Index2Tuple,
    α::CN.NDArray,
    β,
    backend,
)
    return TO.tensorcontract!(
        C, A, pA, conjA, B, pB, conjB, pAB, α, β, backend, TO.DefaultAllocator()
    )
end
function TO.tensorcontract!(
    C::CN.NDArray,
    A::CN.NDArray,
    pA::TO.Index2Tuple,
    conjA::Bool,
    B::CN.NDArray,
    pB::TO.Index2Tuple,
    conjB::Bool,
    pAB::TO.Index2Tuple,
    α::Number,
    β::CN.NDArray,
    backend,
)
    return TO.tensorcontract!(
        C, A, pA, conjA, B, pB, conjB, pAB, α, β, backend, TO.DefaultAllocator()
    )
end
function TO.tensorcontract!(
    C::CN.NDArray,
    A::CN.NDArray,
    pA::TO.Index2Tuple,
    conjA::Bool,
    B::CN.NDArray,
    pB::TO.Index2Tuple,
    conjB::Bool,
    pAB::TO.Index2Tuple,
    α::CN.NDArray,
    β,
    backend,
    allocator,
)
    b = _ensure_backend(TO.tensorcontract!, backend, C, A, B)
    return TO.tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB, α, β, b, allocator)
end
function TO.tensorcontract!(
    C::CN.NDArray,
    A::CN.NDArray,
    pA::TO.Index2Tuple,
    conjA::Bool,
    B::CN.NDArray,
    pB::TO.Index2Tuple,
    conjB::Bool,
    pAB::TO.Index2Tuple,
    α::Number,
    β::CN.NDArray,
    backend,
    allocator,
)
    b = _ensure_backend(TO.tensorcontract!, backend, C, A, B)
    return TO.tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB, α, β, b, allocator)
end

function TO.tensorscalar(C::CN.NDArray)
    ndims(C) == 0 || throw(DimensionMismatch("tensorscalar requires a rank-zero tensor"))
    return copy(C)
end

end
