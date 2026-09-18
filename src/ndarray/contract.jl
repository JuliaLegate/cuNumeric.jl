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
 * Author(s): David Krasowska <krasow@u.northwestern.edu>
 *            Ethan Meitz <emeitz@andrew.cmu.edu>
 *            Nader Rahhal <naderrahhal2026@u.northwestern.edu>
=#

export contract!, contract, tensordot

const _CONTRACT_NATIVE = Union{SUPPORTED_FLOAT_TYPES,SUPPORTED_COMPLEX_TYPES}
const _CONTRACT_PROMOTABLE = Union{SUPPORTED_INT_TYPES,Bool}
const _CONTRACT_ACCEPTED = Union{_CONTRACT_NATIVE,_CONTRACT_PROMOTABLE}

_contract_eltype(::Type{T}) where {T<:_CONTRACT_NATIVE} = T
_contract_eltype(::Type{<:_CONTRACT_PROMOTABLE}) = Float64

function _modes_as_chars(modes::AbstractString)
    chars = Vector{UInt8}(undef, length(modes))
    for (i, c) in enumerate(modes)
        isascii(c) || throw(ArgumentError("contract mode labels must be ASCII, got $(repr(c))"))
        chars[i] = UInt8(c)
    end
    return chars
end

function _modes_as_chars(modes::AbstractVector{UInt8})
    return collect(UInt8, modes)
end

function _modes_as_chars(modes::AbstractVector{Char})
    chars = Vector{UInt8}(undef, length(modes))
    for (i, c) in enumerate(modes)
        isascii(c) || throw(ArgumentError("contract mode labels must be ASCII, got $(repr(c))"))
        chars[i] = UInt8(c)
    end
    return chars
end

function _modes_as_chars(modes::AbstractVector{<:Integer})
    chars = Vector{UInt8}(undef, length(modes))
    for (i, m) in enumerate(modes)
        v = Int(m) + (Int('a') - 1)
        (0 <= v <= 255) || throw(ArgumentError("integer mode label $m is out of range"))
        chars[i] = UInt8(v)
    end
    return chars
end

function _modes_as_chars(modes::Tuple)
    return _modes_as_chars(collect(modes))
end

function _require_modes(arr::NDArray, modes)
    chars = _modes_as_chars(modes)
    length(chars) == ndims(arr) || throw(
        ArgumentError("expected $(ndims(arr)) mode labels, got $(length(chars))")
    )
    if length(Base.unique(chars)) != length(chars)
        throw(ArgumentError("duplicate mode labels are not allowed: $(modes)"))
    end
    return chars
end

function _mode_extents(A::NDArray, Am, B::NDArray, Bm)
    extents = Dict{UInt8,Int}()
    for (m, s) in zip(Am, size(A))
        prev = get(extents, m, s)
        prev == s || throw(
            DimensionMismatch("mode $(Char(m)) has incompatible extents $prev and $s")
        )
        extents[m] = s
    end
    for (m, s) in zip(Bm, size(B))
        prev = get(extents, m, s)
        prev == s || throw(
            DimensionMismatch("mode $(Char(m)) has incompatible extents $prev and $s")
        )
        extents[m] = s
    end
    return extents
end

function _check_mode_counts(Cm, Am, Bm)
    counts = Dict{UInt8,Int}()
    for m in Cm
        counts[m] = get(counts, m, 0) + 1
    end
    for m in Am
        counts[m] = get(counts, m, 0) + 1
    end
    for m in Bm
        counts[m] = get(counts, m, 0) + 1
    end
    for (m, c) in counts
        (c == 2 || c == 3) || throw(
            ArgumentError(
                "mode $(Char(m)) appears $c times; each label must appear twice or three times across the output and inputs"
            ),
        )
    end
    return nothing
end

function _extent_arrays(extents)
    n = length(extents)
    keys_out = Vector{UInt8}(undef, n)
    vals_out = Vector{Int32}(undef, n)
    i = 0
    for (k, v) in extents
        i += 1
        keys_out[i] = k
        vals_out[i] = Int32(v)
    end
    return keys_out, vals_out
end

# cupynumeric's MM shortcut asserts on untransposed NDArray shapes. Present a
# physical `ik,kj->ij` via Legate store transpose (no data copy) when needed.
function _nda_contract!(C, Cm, A, Am, B, Bm, extent_keys, extent_vals)
    if length(Cm) == 2 && length(Am) == 2 && length(Bm) == 2
        i, j = Cm[1], Cm[2]
        k = (Am[1] == i || Am[1] == j) ? Am[2] : Am[1]
        if k != i && k != j
            left, Lm, right, Rm = (i == Am[1] || i == Am[2]) ? (A, Am, B, Bm) : (B, Bm, A, Am)
            oL = Lm[1] != i
            oR = Rm[2] != j
            Lp = oL ? permutedims(left) : left
            Rp = oR ? permutedims(right) : right
            nda_contract(C, Cm, Lp, UInt8[i, k], Rp, UInt8[k, j], extent_keys, extent_vals)
            oL && destroy!(Lp)
            oR && destroy!(Rp)
            return nothing
        end
    end
    nda_contract(C, Cm, A, Am, B, Bm, extent_keys, extent_vals)
    return nothing
end

function _free_output_modes(A::NDArray, Am, B::NDArray, Bm)
    counts = Dict{UInt8,Int}()
    for m in Am
        counts[m] = get(counts, m, 0) + 1
    end
    for m in Bm
        counts[m] = get(counts, m, 0) + 1
    end
    Cm = UInt8[]
    cshape = Int[]
    for (m, s) in zip(Am, size(A))
        if counts[m] == 1
            push!(Cm, m)
            push!(cshape, s)
        end
    end
    for (m, s) in zip(Bm, size(B))
        if counts[m] == 1
            push!(Cm, m)
            push!(cshape, s)
        end
    end
    return Cm, Tuple(cshape)
end

"""
    contract!(C, Cmodes, A, Amodes, B, Bmodes; α=1, β=0)

In-place pairwise tensor contraction `C = β * C + α * (A ⋆ B)`.

`α` and `β` may be a Julia `Number` or a 0-d `NDArray`.

`Amodes`, `Bmodes`, and `Cmodes` are mode labels for `A`, `B`, and `C`: an
ASCII `AbstractString`, a tuple/vector of `Char`, or a vector of integers
(`1` maps to `'a'`). Each label appears twice (a contracted or free index) or
three times (a batched / Hadamard index). Duplicate labels inside one array
are not allowed — extract a diagonal first with `cuNumeric.diagonal`.

Supported element types are `Float32`, `Float64`, `ComplexF32`, and
`ComplexF64`. Integer and `Bool` inputs are converted to `Float64` under the
usual promotion rules.

This is the pairwise primitive TensorOperations.jl can call later. It does not
parse einsum strings. Multi-GPU execution uses Legate tiling (not cuTensorMp).
"""
function contract!(
    C::NDArray{TC},
    Cmodes,
    A::NDArray{TA},
    Amodes,
    B::NDArray{TB},
    Bmodes;
    α=1,
    β=0,
) where {TC<:_CONTRACT_NATIVE,TA<:_CONTRACT_ACCEPTED,TB<:_CONTRACT_ACCEPTED}
    T = promote_type(_contract_eltype(TA), _contract_eltype(TB))
    T === TC || throw(
        ArgumentError("contract! output has type $TC, but inputs promote to $T")
    )

    Ap = checked_promote_arr(contract!, A, T)
    Bp = checked_promote_arr(contract!, B, T)
    try
        return _contract_same_type!(C, Cmodes, Ap, Amodes, Bp, Bmodes, α, β)
    finally
        Ap !== A && destroy!(Ap)
        Bp !== B && destroy!(Bp)
    end
end

function contract!(C::NDArray, Cmodes, A::NDArray, Amodes, B::NDArray, Bmodes; α=1, β=0)
    bad = if eltype(C) <: _CONTRACT_NATIVE
        (eltype(A) <: _CONTRACT_ACCEPTED ? eltype(B) : eltype(A))
    else
        eltype(C)
    end
    return throw(ArgumentError("array type $bad is unsupported in contract!"))
end

function _require_0d_scale(x::NDArray)
    ndims(x) == 0 || throw(
        ArgumentError("contract! scale factor must be a Number or a 0-d NDArray")
    )
    return x
end

function _contract_prepare(
    C::NDArray{T}, Cmodes, A::NDArray{T}, Amodes, B::NDArray{T}, Bmodes
) where {T}
    (C.ptr === A.ptr || C.ptr === B.ptr) && throw(
        ArgumentError("contract! output must not alias either input")
    )

    Cm = _require_modes(C, Cmodes)
    Am = _require_modes(A, Amodes)
    Bm = _require_modes(B, Bmodes)
    _check_mode_counts(Cm, Am, Bm)
    extents = _mode_extents(A, Am, B, Bm)
    for (m, s) in zip(Cm, size(C))
        expected = get(extents, m, -1)
        expected == s || throw(
            DimensionMismatch("output mode $(Char(m)) has size $s, expected $expected")
        )
    end
    extent_keys, extent_vals = _extent_arrays(extents)
    return Cm, Am, Bm, extent_keys, extent_vals
end

function _contract_same_type!(
    C::NDArray{T},
    Cmodes,
    A::NDArray{T},
    Amodes,
    B::NDArray{T},
    Bmodes,
    α::Number,
    β::Number,
) where {T}
    Cm, Am, Bm, extent_keys, extent_vals = _contract_prepare(C, Cmodes, A, Amodes, B, Bmodes)
    if isone(α) && iszero(β)
        _nda_contract!(C, Cm, A, Am, B, Bm, extent_keys, extent_vals)
        return C
    end
    αT = convert(T, α)
    if iszero(β)
        _nda_contract!(C, Cm, A, Am, B, Bm, extent_keys, extent_vals)
        C .= αT .* C
        return C
    end
    βT = convert(T, β)
    tmp = similar(C)
    _nda_contract!(tmp, Cm, A, Am, B, Bm, extent_keys, extent_vals)
    C .= βT .* C .+ αT .* tmp
    destroy!(tmp)
    return C
end

function _contract_same_type!(
    C::NDArray{T},
    Cmodes,
    A::NDArray{T},
    Amodes,
    B::NDArray{T},
    Bmodes,
    α::NDArray,
    β::Number,
) where {T}
    _require_0d_scale(α)
    Cm, Am, Bm, extent_keys, extent_vals = _contract_prepare(C, Cmodes, A, Amodes, B, Bmodes)
    α′ = eltype(α) === T ? α : as_type(α, T)
    try
        if iszero(β)
            _nda_contract!(C, Cm, A, Am, B, Bm, extent_keys, extent_vals)
            C .= α′ .* C
            return C
        end
        tmp = similar(C)
        _nda_contract!(tmp, Cm, A, Am, B, Bm, extent_keys, extent_vals)
        if isone(β)
            C .= C .+ α′ .* tmp
        else
            βa = NDArray(convert(T, β))
            C .= βa .* C .+ α′ .* tmp
            destroy!(βa)
        end
        destroy!(tmp)
        return C
    finally
        α′ !== α && destroy!(α′)
    end
end

function _contract_same_type!(
    C::NDArray{T},
    Cmodes,
    A::NDArray{T},
    Amodes,
    B::NDArray{T},
    Bmodes,
    α::Number,
    β::NDArray,
) where {T}
    _require_0d_scale(β)
    αa = NDArray(convert(T, α))
    try
        return _contract_same_type!(C, Cmodes, A, Amodes, B, Bmodes, αa, β)
    finally
        destroy!(αa)
    end
end

function _contract_same_type!(
    C::NDArray{T},
    Cmodes,
    A::NDArray{T},
    Amodes,
    B::NDArray{T},
    Bmodes,
    α::NDArray,
    β::NDArray,
) where {T}
    _require_0d_scale(α)
    _require_0d_scale(β)
    Cm, Am, Bm, extent_keys, extent_vals = _contract_prepare(C, Cmodes, A, Amodes, B, Bmodes)
    α′ = eltype(α) === T ? α : as_type(α, T)
    β′ = eltype(β) === T ? β : as_type(β, T)
    try
        tmp = similar(C)
        _nda_contract!(tmp, Cm, A, Am, B, Bm, extent_keys, extent_vals)
        C .= β′ .* C .+ α′ .* tmp
        destroy!(tmp)
        return C
    finally
        α′ !== α && destroy!(α′)
        β′ !== β && destroy!(β′)
    end
end

"""
    contract(A, Amodes, B, Bmodes; α=1)

Allocate and return `α * (A ⋆ B)`. Output modes are the labels that appear
once, in the order they occur on `A` then `B` (classical Einstein). For a
batched or Hadamard product, allocate `C` yourself and call [`contract!`](@ref)
with explicit `Cmodes`.
"""
function contract(
    A::NDArray{TA}, Amodes, B::NDArray{TB}, Bmodes; α=1
) where {TA<:_CONTRACT_ACCEPTED,TB<:_CONTRACT_ACCEPTED}
    T = promote_type(_contract_eltype(TA), _contract_eltype(TB))
    T <: _CONTRACT_NATIVE ||
        throw(ArgumentError("array type $T is unsupported in contract"))

    Am = _require_modes(A, Amodes)
    Bm = _require_modes(B, Bmodes)
    Cm, cshape = _free_output_modes(A, Am, B, Bm)
    C = cuNumeric.zeros(T, cshape)
    return contract!(C, Cm, A, Am, B, Bm; α=α, β=zero(T))
end

function contract(A::NDArray, Amodes, B::NDArray, Bmodes; α=1)
    bad = eltype(A) <: _CONTRACT_ACCEPTED ? eltype(B) : eltype(A)
    return throw(ArgumentError("array type $bad is unsupported in contract"))
end

"""
    tensordot(A, B, axes=2; α=1)
    tensordot(A, B, (a_axes, b_axes); α=1)

Contract `A` with `B` along the given 1-based axes. `axes::Integer` contracts
the last `axes` dimensions of `A` with the first `axes` of `B`. A tuple of axis
collections names the axes on each input. Remaining axes of `A` then `B` become
the output.
"""
function tensordot(A::NDArray, B::NDArray, axes::Integer=2; α=1)
    n = Int(axes)
    n < 0 && throw(ArgumentError("axes must be non-negative, got $n"))
    na = ndims(A)
    nb = ndims(B)
    n > na && throw(ArgumentError("cannot contract $n axes of a $(na)-d array"))
    n > nb && throw(ArgumentError("cannot contract $n axes of a $(nb)-d array"))
    a_axes = ntuple(i -> na - n + i, n)
    b_axes = ntuple(identity, n)
    return tensordot(A, B, (a_axes, b_axes); α=α)
end

function tensordot(A::NDArray, B::NDArray, axes::Tuple; α=1)
    a_raw, b_raw = axes
    a_axes = collect(Int, a_raw isa Integer ? (a_raw,) : a_raw)
    b_axes = collect(Int, b_raw isa Integer ? (b_raw,) : b_raw)
    length(a_axes) == length(b_axes) ||
        throw(ArgumentError("tensordot axis lists must have the same length"))
    length(Base.unique(a_axes)) == length(a_axes) ||
        throw(ArgumentError("duplicate axes on first input: $a_axes"))
    length(Base.unique(b_axes)) == length(b_axes) ||
        throw(ArgumentError("duplicate axes on second input: $b_axes"))

    na = ndims(A)
    nb = ndims(B)
    Am = Vector{UInt8}(undef, na)
    Bm = Vector{UInt8}(undef, nb)
    for i in 1:na
        Am[i] = UInt8('a' + (i - 1))
    end
    for i in 1:nb
        Bm[i] = UInt8('A' + (i - 1))
    end
    for (ai, bi) in zip(a_axes, b_axes)
        (1 <= ai <= na) || throw(ArgumentError("axis $ai is out of range for $(na)-d array"))
        (1 <= bi <= nb) || throw(ArgumentError("axis $bi is out of range for $(nb)-d array"))
        size(A, ai) == size(B, bi) || throw(
            DimensionMismatch(
                "tensordot axes $ai and $bi have sizes $(size(A, ai)) and $(size(B, bi))"
            ),
        )
        Bm[bi] = Am[ai]
    end
    return contract(A, Am, B, Bm; α=α)
end
