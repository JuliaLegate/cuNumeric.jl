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

const SORT_TYPES = (Bool, Base.uniontypes(cuNumeric.SUPPORTED_NUMERIC_TYPES)...)

# Julia has no isless(::Complex, ::Complex). Use this only as Base.sort's `by`
# so the reference matches cupynumeric's (real, imag) order.
_lex_complex(x) = (real(x), imag(x))

function _sort_fixture(::Type{T}) where {T}
    T <: Bool && return Bool[1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1]
    T <: Complex &&
        return T[
            10 + 3im,
            2 + 4im,
            1 + 5im,
            8 + 9im,
            3 + 1im,
            2 + 4im,
            10 + 3im,
            1,
            5 + 2im,
            0,
            4,
            7 + im,
        ]
    return T[10, 3, 12, 5, 2, 4, 8, 9, 7, 6, 11, 1]
end

function _dup_fixture(::Type{T}) where {T}
    T <: Bool && return Bool[1, 0, 1, 0, 0, 1]
    T <: Complex && return T[10 + 3im, 2 + 4im, 10 + 3im, 2 + 4im, 1, 1]
    return T[10, 3, 12, 5, 2, 3, 8, 8, 7, 6, 10, 1]
end

function _search_haystack(::Type{T}) where {T}
    T <: Bool && return Bool[0, 0, 1, 1]
    return T[1, 2, 2, 4, 5, 7]
end

function _search_needles(::Type{T}) where {T}
    T <: Bool && return Bool[0, 1]
    return T[0, 1, 2, 3, 7, 8]
end

function _unique_fixture(::Type{T}) where {T}
    T <: Bool && return Bool[1, 0, 1, 0, 1]
    T <: Complex && return T[1, 2, 2, 3 + im, 3 + im, 1]
    return T[1, 2, 2, 3, 4, 4, 4, 5]
end

@testset "sort 1-d" begin
    @testset verbose = true for T in SORT_TYPES
        A = _sort_fixture(T)
        nda = cuNumeric.NDArray(A)
        @test Array(cuNumeric.sort(nda)) ==
            (T <: Complex ? Base.sort(A; by=_lex_complex) : Base.sort(A))
        @test Array(nda) == A  # cuNumeric.sort is not in-place
    end
end

@testset "sort! 1-d" begin
    @testset verbose = true for T in SORT_TYPES
        A = _sort_fixture(T)
        nda = cuNumeric.NDArray(copy(A))
        cuNumeric.sort!(nda)
        @test Array(nda) == (T <: Complex ? Base.sort(A; by=_lex_complex) : Base.sort(A))
    end
end

@testset "sort dims" begin
    @testset verbose = true for T in SORT_TYPES
        A = reshape(_sort_fixture(T), 3, 4)
        nda = cuNumeric.NDArray(A)
        @test Array(cuNumeric.sort(nda; dims=1)) ==
            (T <: Complex ? Base.sort(A; dims=1, by=_lex_complex) : Base.sort(A; dims=1))
        @test Array(cuNumeric.sort(nda; dims=2)) ==
            (T <: Complex ? Base.sort(A; dims=2, by=_lex_complex) : Base.sort(A; dims=2))
        @test_throws "invalid for a" cuNumeric.sort(nda; dims=3)
        @test_throws "invalid for a" cuNumeric.sort(nda; dims=0)
        @test_throws UndefKeywordError cuNumeric.sort(nda)  # dims required for N>1
    end
end

@testset "sortperm unique" begin
    @testset verbose = true for T in SORT_TYPES
        A = _sort_fixture(T)
        nda = cuNumeric.NDArray(A)
        @test Array(cuNumeric.sortperm(nda)) ==
            (T <: Complex ? Base.sortperm(A; by=_lex_complex) : Base.sortperm(A))
        B = reshape(A, 3, 4)
        ndb = cuNumeric.NDArray(B)
        p1 = Array(cuNumeric.sortperm(ndb; dims=1))
        p2 = Array(cuNumeric.sortperm(ndb; dims=2))
        @test p1 ==
            (T <: Complex ? Base.sortperm(B; dims=1, by=_lex_complex) : Base.sortperm(B; dims=1))
        @test p2 ==
            (T <: Complex ? Base.sortperm(B; dims=2, by=_lex_complex) : Base.sortperm(B; dims=2))
        @test B[p1] ==
            (T <: Complex ? Base.sort(B; dims=1, by=_lex_complex) : Base.sort(B; dims=1))
        @test B[p2] ==
            (T <: Complex ? Base.sort(B; dims=2, by=_lex_complex) : Base.sort(B; dims=2))
    end
end

@testset "sortperm stable duplicates" begin
    @testset verbose = true for T in SORT_TYPES
        A = _dup_fixture(T)
        nda = cuNumeric.NDArray(A)
        @test Array(cuNumeric.sortperm(nda; stable=true)) == (
            if T <: Complex
                Base.sortperm(A; alg=Base.MergeSort, by=_lex_complex)
            else
                Base.sortperm(A; alg=Base.MergeSort)
            end
        )
        @test Array(cuNumeric.sort(nda; stable=true)) ==
            (T <: Complex ? Base.sort(A; by=_lex_complex) : Base.sort(A))
    end
end

@testset "unsupported kwargs and not Base.sort" begin
    v = cuNumeric.NDArray(Int32[3, 1, 2])
    @test_throws MethodError cuNumeric.sort(v; alg=Base.QuickSort)
    @test_throws MethodError cuNumeric.sort(v; rev=true)
    @test_throws MethodError cuNumeric.sort(v; lt=(!))
    # cuNumeric.sort is not Base.sort: Julia arrays are a MethodError
    @test_throws MethodError cuNumeric.sort([3, 1, 2])
    @test_throws MethodError cuNumeric.sort!([3, 1, 2])
    @test_throws MethodError cuNumeric.sortperm([3, 1, 2])
    @test_throws MethodError cuNumeric.searchsortedfirst([1, 2, 3], 2)
    @test_throws MethodError cuNumeric.unique([1, 1, 2])
end

@testset "searchsorted" begin
    @testset verbose = true for T in SORT_TYPES
        A = _search_haystack(T)
        nda = cuNumeric.sort(cuNumeric.NDArray(A))
        if T <: Complex
            @test_throws "not supported for complex" cuNumeric.searchsortedfirst(nda, zero(T))
            @test_throws "not supported for complex" cuNumeric.searchsortedlast(nda, zero(T))
            @test_throws "not supported for complex" cuNumeric.searchsorted(nda, zero(T))
            continue
        end
        for x in _search_needles(T)
            @test cuNumeric.unwrap(cuNumeric.searchsortedfirst(nda, x)) ==
                Base.searchsortedfirst(A, x)
            @test cuNumeric.unwrap(cuNumeric.searchsortedlast(nda, x)) ==
                Base.searchsortedlast(A, x)
            @test cuNumeric.searchsorted(nda, x) == Base.searchsorted(A, x)
        end
        needles = _search_needles(T)
        firsts = Array(cuNumeric.searchsortedfirst(nda, cuNumeric.NDArray(needles)))
        lasts = Array(cuNumeric.searchsortedlast(nda, cuNumeric.NDArray(needles)))
        @test firsts == Base.searchsortedfirst.(Ref(A), needles)
        @test lasts == Base.searchsortedlast.(Ref(A), needles)
        M = cuNumeric.NDArray(reshape(A, 2, :))
        @test_throws MethodError cuNumeric.searchsortedfirst(M, zero(T))
    end
end

@testset "unique" begin
    @testset verbose = true for T in SORT_TYPES
        A = _unique_fixture(T)
        out = Array(cuNumeric.unique(cuNumeric.NDArray(A)))
        @test Set(out) == Set(Base.unique(A))
        if !(T <: Complex)
            @test Base.issorted(out)
        end
        B = reshape(_sort_fixture(T), 3, 4)
        @test Set(Array(cuNumeric.unique(cuNumeric.NDArray(B)))) == Set(Base.unique(vec(B)))
    end
    @test_throws MethodError cuNumeric.unique(cuNumeric.NDArray(Int32[1, 1]); dims=1)
end
