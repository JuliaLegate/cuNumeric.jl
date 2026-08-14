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

# Julia 1-based dims -> 0-based cupynumeric axis.
function _checked_sort_axis(N::Int, dims::Integer)
    1 <= dims <= N || throw(ArgumentError("dims=$dims is invalid for a $N-d array"))
    return Int32(dims - 1)
end

@doc"""
    cuNumeric.sort(v::NDArray{T,1}; stable::Bool=false)
    cuNumeric.sort(A::NDArray; dims::Integer, stable::Bool=false)

Return a copy of `A` sorted in ascending order.

This is **not** `Base.sort`. Call it as `cuNumeric.sort`. `alg`, `lt`, `by`,
`rev`, and `order` are not accepted.

For rank greater than 1, `dims` is required (Julia 1-based). `stable=true`
sets the cupynumeric stable-sort flag (`kind="stable"`); the default is
unstable (`kind="quicksort"`). These names do not run Julia's QuickSort or
MergeSort. Complex values are ordered lexicographically by `(real, imag)`.
"""
function sort(arr::NDArray{T,1}; stable::Bool=false) where {T}
    return nda_sort(arr, Int32(-1), stable)
end

function sort(arr::NDArray{T,N}; dims::Integer, stable::Bool=false) where {T,N}
    return nda_sort(arr, _checked_sort_axis(N, dims), stable)
end

@doc"""
    cuNumeric.sort!(v::NDArray{T,1}; stable::Bool=false)
    cuNumeric.sort!(A::NDArray; dims::Integer, stable::Bool=false)

Sort `A` in place. Same kwargs as [`cuNumeric.sort`](@ref). Not `Base.sort!`.
"""
function sort!(arr::NDArray{T,1}; stable::Bool=false) where {T}
    return nda_sort_inplace(arr, Int32(-1), stable)
end

function sort!(arr::NDArray{T,N}; dims::Integer, stable::Bool=false) where {T,N}
    return nda_sort_inplace(arr, _checked_sort_axis(N, dims), stable)
end

# cupynumeric argsort is along-axis (like NumPy). Julia's sortperm(; dims)
# stores linear indices so that `A[sortperm(A; dims)] == sort(A; dims)`.
function _along_axis_to_linear(along::NDArray{Int64,N}, dims::Integer) where {N}
    idx = Array(along)
    destroy!(along)
    s = strides(idx)[Int(dims)]
    lin = LinearIndices(idx)
    result = similar(idx)
    @inbounds for I in CartesianIndices(idx)
        result[I] = lin[I] + (idx[I] - I[dims]) * s
    end
    return NDArray(result)
end

@doc"""
    cuNumeric.sortperm(v::NDArray{T,1}; stable::Bool=false)
    cuNumeric.sortperm(A::NDArray; dims::Integer, stable::Bool=false)

1-based permutation indices as an `NDArray{Int64}` of the same shape as `A`.
For rank greater than 1, the entries are linear indices into `A`, matching
`Base.sortperm(; dims)`. Not `Base.sortperm`. Same `dims` / `stable` kwargs
as [`cuNumeric.sort`](@ref).
"""
function sortperm(arr::NDArray{T,1}; stable::Bool=false) where {T}
    return _indices_to_one_based(nda_argsort(arr, Int32(-1), stable))
end

function sortperm(arr::NDArray{T,N}; dims::Integer, stable::Bool=false) where {T,N}
    along = _indices_to_one_based(nda_argsort(arr, _checked_sort_axis(N, dims), stable))
    return _along_axis_to_linear(along, dims)
end

function _searchsorted_impl(a::NDArray{TA,1}, v::NDArray{TV,N}, left::Bool) where {TA,TV,N}
    (TA <: Complex || TV <: Complex) &&
        throw(ArgumentError("searchsorted is not supported for complex arrays"))
    U = promote_type(TA, TV)
    a2 = unchecked_promote_arr(a, U)
    v2 = unchecked_promote_arr(v, U)
    raw = nda_searchsorted(a2, v2, left)
    a2 !== a && destroy!(a2)
    v2 !== v && destroy!(v2)
    return left ? _indices_to_one_based(raw) : raw
end

@doc"""
    cuNumeric.searchsortedfirst(a::NDArray{T,1}, x)
    cuNumeric.searchsortedlast(a::NDArray{T,1}, x)

Insertion indices into a 1-d sorted `a`. `x` may be a `Number` or an
`NDArray` of needles.

Scalar queries return a 0-d `NDArray{Int64}` (not a Julia `Int`); use
`unwrap` for an `Int`. Array queries return an `NDArray{Int64}` with the
shape of the needles. Indices are 1-based.

Not `Base.searchsortedfirst` / `searchsortedlast`. `a` must already be sorted
ascending. `lt` / `by` / `rev` are not accepted. Complex arrays are not
supported.
"""
function searchsortedfirst(a::NDArray{T,1}, v::NDArray) where {T}
    return _searchsorted_impl(a, v, true)
end

function searchsortedfirst(a::NDArray{T,1}, x::Number) where {T}
    needle = NDArray(convert(T, x))
    result = searchsortedfirst(a, needle)
    destroy!(needle)
    return result
end

function searchsortedlast(a::NDArray{T,1}, v::NDArray) where {T}
    return _searchsorted_impl(a, v, false)
end

function searchsortedlast(a::NDArray{T,1}, x::Number) where {T}
    needle = NDArray(convert(T, x))
    result = searchsortedlast(a, needle)
    destroy!(needle)
    return result
end

@doc"""
    cuNumeric.searchsorted(a::NDArray{T,1}, x::Number)

`searchsortedfirst(a, x):searchsortedlast(a, x)` as a `UnitRange`, matching
Base's scalar search. Materializes two 0-d index arrays via `unwrap`.
Not `Base.searchsorted`.
"""
function searchsorted(a::NDArray{T,1}, x::Number) where {T}
    lo = searchsortedfirst(a, x)
    hi = searchsortedlast(a, x)
    return unwrap(lo):unwrap(hi)
end

@doc"""
    cuNumeric.unique(A::NDArray) -> NDArray{T,1}

Sorted unique elements of `A`, flattened to 1-d.

This is **not** `Base.unique`, which keeps first-occurrence order and does
not sort. `dims`, `return_index`, `return_inverse`, and `return_counts`
are not accepted; cupynumeric does not implement them.
"""
function unique(arr::NDArray)
    return nda_unique(arr)
end
