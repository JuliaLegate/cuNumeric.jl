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

using TensorOperations

# Contractions sum products; TBLIS/cuTENSOR/BLAS may associate differently than
# the host ref. Default (`cancel=true`): Higham floor like unary reductions
# (`n` = contracted length, `scale` ≈ max|A| * max|B|). All-positive inputs
# set `cancel=false` and drop that floor. No-sum cases keep n=1.
function _host_contract_compare(
    ref, out, ::Type{T}; n::Integer=1, scale=1, cancel::Bool=true
) where {T}
    atolv, rtolv = if n > 1 && cancel
        reduction_atol(T, n, scale), reduction_rtol(T, n)
    elseif n > 1
        atol(T) * n, reduction_rtol(T, n)
    else
        atol(T), rtol(T)
    end
    allowscalar() do
        @test safe_compare(ref, out, atolv, rtolv)
    end
end

_contract_scale(A, B) = maximum(abs, A) * maximum(abs, B)

# TensorOperations forbids an index on both inputs and the output (batched /
# Hadamard). Those refs are explicit loops or broadcasting.
function _batched_ref(A::Array{T,3}, B::Array{T,3}) where {T}
    ni, nj, nk = size(A)
    nl = size(B, 3)
    C = zeros(T, ni, nj, nl)
    for i in 1:ni, j in 1:nj, l in 1:nl
        s = zero(T)
        for k in 1:nk
            s += A[i, j, k] * B[i, k, l]
        end
        C[i, j, l] = s
    end
    return C
end

@testset "contract GEMM" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        A = my_rand(T, 5, 4)
        B = my_rand(T, 4, 6)
        nda = NDArray(A)
        ndb = NDArray(B)
        nk = size(A, 2)
        scale = _contract_scale(A, B)
        @tensor ref_ij[i, j] := A[i, k] * B[k, j]
        @tensor ref_ji[j, i] := A[i, k] * B[k, j]

        C = contract(nda, "ik", ndb, "kj")
        @test size(C) == (5, 6)
        _host_contract_compare(ref_ij, C, T; n=nk, scale)

        out = cuNumeric.zeros(T, 5, 6)
        contract!(out, "ij", nda, "ik", ndb, "kj")
        _host_contract_compare(ref_ij, out, T; n=nk, scale)

        C_int = contract(nda, (1, 2), ndb, (2, 3))
        _host_contract_compare(ref_ij, C_int, T; n=nk, scale)

        # Every MM layout: A/B axis order, operand swap, output permutation.
        ndaT = permutedims(nda)
        ndbT = permutedims(ndb)
        for (Ause, Am) in ((nda, "ik"), (ndaT, "ki"))
            for (Buse, Bm) in ((ndb, "kj"), (ndbT, "jk"))
                for swap in (false, true)
                    X, Xm, Y, Ym = swap ? (Buse, Bm, Ause, Am) : (Ause, Am, Buse, Bm)
                    Cij = cuNumeric.zeros(T, 5, 6)
                    contract!(Cij, "ij", X, Xm, Y, Ym)
                    _host_contract_compare(ref_ij, Cij, T; n=nk, scale)
                    Cji = cuNumeric.zeros(T, 6, 5)
                    contract!(Cji, "ji", X, Xm, Y, Ym)
                    _host_contract_compare(ref_ji, Cji, T; n=nk, scale)
                end
            end
        end
    end
end

@testset "contract MV / VV / outer" begin
    @testset for T in (Float32, Float64)
        A = my_rand(T, 5, 4)
        x = my_rand(T, 4)
        y = my_rand(T, 5)
        nda = NDArray(A)
        ndx = NDArray(x)
        ndy = NDArray(y)

        @tensor ref_mv[i] := A[i, j] * x[j]
        _host_contract_compare(
            ref_mv, contract(nda, "ij", ndx, "j"), T; n=length(x), scale=_contract_scale(A, x)
        )
        _host_contract_compare(
            ref_mv, contract(ndx, "j", nda, "ij"), T; n=length(x), scale=_contract_scale(A, x)
        )

        @tensor ref_vm[i] := A[j, i] * y[j]
        _host_contract_compare(
            ref_vm, contract(nda, "ji", ndy, "j"), T; n=length(y), scale=_contract_scale(A, y)
        )
        _host_contract_compare(
            ref_vm, contract(ndy, "j", nda, "ji"), T; n=length(y), scale=_contract_scale(A, y)
        )

        u = my_rand(T, 6)
        v = my_rand(T, 6)
        @tensor ref_dot[] := u[i] * v[i]
        dot = contract(NDArray(u), "i", NDArray(v), "i")
        @test ndims(dot) == 0
        _host_contract_compare(ref_dot, dot, T; n=length(u), scale=_contract_scale(u, v))

        @tensor ref_outer[i, j] := u[i] * v[j]
        outer = contract(NDArray(u), "i", NDArray(v), "j")
        @test size(outer) == (6, 6)
        _host_contract_compare(ref_outer, outer, T)
    end
end

@testset "contract batched" begin
    @testset for T in (Float32, Float64)
        A = my_rand(T, 3, 4, 5)
        B = my_rand(T, 3, 5, 6)
        nda = NDArray(A)
        ndb = NDArray(B)
        ref = _batched_ref(A, B)
        nk = size(A, 3)
        scale = _contract_scale(A, B)
        # All 6 output orders of (i, j, l)
        for (cm, p) in (
            ("ijl", (1, 2, 3)),
            ("ilj", (1, 3, 2)),
            ("jil", (2, 1, 3)),
            ("jli", (2, 3, 1)),
            ("lij", (3, 1, 2)),
            ("lji", (3, 2, 1)),
        )
            C = cuNumeric.zeros(T, map(d -> size(ref, d), p)...)
            contract!(C, cm, nda, "ijk", ndb, "ikl")
            _host_contract_compare(permutedims(ref, p), C, T; n=nk, scale)
        end
        # Input axis orders (general path, not MM)
        C = cuNumeric.zeros(T, 3, 4, 6)
        contract!(C, "ijl", permutedims(nda, (2, 1, 3)), "jik", ndb, "ikl")
        _host_contract_compare(ref, C, T; n=nk, scale)
        contract!(C, "ijl", nda, "ijk", permutedims(ndb, (2, 1, 3)), "kil")
        _host_contract_compare(ref, C, T; n=nk, scale)
    end
end

@testset "contract Hadamard" begin
    @testset for T in (Float32, Float64)
        A = my_rand(T, 4, 5)
        B = my_rand(T, 4, 5)
        ref = A .* B
        C = cuNumeric.zeros(T, 4, 5)
        contract!(C, "ij", NDArray(A), "ij", NDArray(B), "ij")
        _host_contract_compare(ref, C, T)
        Cji = cuNumeric.zeros(T, 5, 4)
        contract!(Cji, "ji", NDArray(A), "ij", NDArray(B), "ij")
        _host_contract_compare(permutedims(ref), Cji, T)
    end
end

@testset "contract alpha/beta" begin
    @testset for T in (Float32, Float64)
        A = my_rand(T, 4, 3)
        B = my_rand(T, 3, 5)
        nda = NDArray(A)
        ndb = NDArray(B)
        α2, α3, β3 = T(2), T(3), T(3)
        nk = size(A, 2)
        scale = _contract_scale(A, B)
        @tensor prod[i, j] := A[i, k] * B[k, j]

        C = contract(nda, "ik", ndb, "kj"; α=α2)
        _host_contract_compare(α2 * prod, C, T; n=nk, scale=α2 * scale)

        out = cuNumeric.zeros(T, 4, 5)
        contract!(out, "ij", nda, "ik", ndb, "kj"; α=α3, β=0)
        _host_contract_compare(α3 * prod, out, T; n=nk, scale=α3 * scale)

        seed = my_rand(T, 4, 5)
        out = NDArray(copy(seed))
        contract!(out, "ij", nda, "ik", ndb, "kj"; α=α2, β=β3)
        _host_contract_compare(β3 * seed + α2 * prod, out, T; n=nk, scale=α2 * scale)

        # α/β on a non-canonical MM layout
        out = NDArray(copy(seed))
        contract!(out, "ij", permutedims(nda), "ki", ndb, "kj"; α=α2, β=β3)
        _host_contract_compare(β3 * seed + α2 * prod, out, T; n=nk, scale=α2 * scale)

        seed_ji = permutedims(seed)
        out_ji = NDArray(copy(seed_ji))
        contract!(out_ji, "ji", nda, "ik", ndb, "kj"; α=α2, β=β3)
        _host_contract_compare(
            β3 * seed_ji + α2 * permutedims(prod), out_ji, T; n=nk, scale=α2 * scale
        )
    end
end

@testset "tensordot" begin
    @testset for T in (Float32, Float64)
        A = my_rand(T, 3, 4, 5)
        B = my_rand(T, 5, 6)
        nda = NDArray(A)
        ndb = NDArray(B)
        @tensor ref[i, j, l] := A[i, j, k] * B[k, l]

        C = tensordot(nda, ndb, 1)
        @test size(C) == (3, 4, 6)
        _host_contract_compare(ref, C, T; n=size(A, 3), scale=_contract_scale(A, B))

        D = tensordot(nda, ndb, ([3], [1]))
        _host_contract_compare(ref, D, T; n=size(A, 3), scale=_contract_scale(A, B))
    end
end

@testset "contract no cancellation" begin
    @testset for T in (Float32, Float64)
        A = my_rand(T, 5, 4; L=one(T), R=T(1000))
        B = my_rand(T, 4, 6; L=one(T), R=T(1000))
        nk = size(A, 2)
        @tensor ref[i, j] := A[i, k] * B[k, j]
        _host_contract_compare(
            ref, contract(NDArray(A), "ik", NDArray(B), "kj"), T; n=nk, cancel=false
        )

        A3 = my_rand(T, 3, 4, 5; L=one(T), R=T(1000))
        B3 = my_rand(T, 3, 5, 6; L=one(T), R=T(1000))
        C = cuNumeric.zeros(T, 3, 4, 6)
        contract!(C, "ijl", NDArray(A3), "ijk", NDArray(B3), "ikl")
        _host_contract_compare(_batched_ref(A3, B3), C, T; n=size(A3, 3), cancel=false)
    end
end

@testset "contract errors" begin
    A = cuNumeric.ones(Float32, 3, 4)
    B = cuNumeric.ones(Float32, 4, 5)
    @test_throws ArgumentError contract(A, "ii", B, "jk")
    @test_throws ArgumentError contract(A, "ik", B, "kjx")
    @test_throws DimensionMismatch contract(A, "ik", cuNumeric.ones(Float32, 3, 5), "kj")
    C = cuNumeric.zeros(Float64, 3, 5)
    @test_throws ArgumentError contract!(C, "ij", A, "ik", B, "kj")
    @test_throws ArgumentError contract!(A, "ij", A, "ik", B, "kj")
end
