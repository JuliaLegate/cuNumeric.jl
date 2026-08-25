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

using TensorOperations
using TensorOperations: TensorOperations as TO

@testset "TensorOperations extension" begin
    @test Base.get_extension(cuNumeric, :cuNumericTensorOperationsExt) !== nothing

    @testset "@tensor permutations and accumulation" begin
        hostA = reshape(collect(Float64, 1:24), 2, 3, 4)
        hostB = reshape(collect(Float64, 1:24), 3, 4, 2)
        A = NDArray(hostA)
        B = NDArray(hostB)

        @tensor permuted[k, i, j] := A[i, j, k]
        @test permuted isa NDArray
        @test Array(permuted) == permutedims(hostA, (3, 1, 2))

        @tensor combined[k, i, j] := 2 * A[i, j, k] - 3 * B[j, k, i]
        ref = 2 .* permutedims(hostA, (3, 1, 2)) .-
              3 .* permutedims(hostB, (2, 3, 1))
        @test Array(combined) == ref

        accumulated = cuNumeric.zeros(Float64, 4, 2, 3)
        @tensor accumulated[k, i, j] = A[i, j, k]
        @tensor accumulated[k, i, j] += 2 * B[j, k, i]
        @tensor accumulated[k, i, j] -= 3 * A[i, j, k]
        @test Array(accumulated) ==
            -2 .* permutedims(hostA, (3, 1, 2)) .+
              2 .* permutedims(hostB, (2, 3, 1))
    end

    @testset "@tensor traces and output permutations" begin
        host = reshape(collect(Float64, 1:(2 * 3 * 4 * 3 * 5 * 4)), 2, 3, 4, 3, 5, 4)
        A = NDArray(host)

        @tensor traced[d, a] := A[a, b, c, b, d, c]
        @tensor ref[d, a] := host[a, b, c, b, d, c]
        @test traced isa NDArray
        @test Array(traced) == ref

        destination = cuNumeric.ones(Float64, 5, 2)
        @tensor destination[d, a] += 0.5 * A[a, b, c, b, d, c]
        @test Array(destination) == ones(5, 2) .+ 0.5 .* ref
    end

    @testset "@tensor contractions and outer products" begin
        hostA = reshape(collect(Float64, 1:(2 * 3 * 4 * 5)), 2, 3, 4, 5)
        hostB = reshape(collect(Float64, 1:(4 * 6 * 3 * 7)), 4, 6, 3, 7)
        A = NDArray(hostA)
        B = NDArray(hostB)

        @tensor contracted[a, g, d, f] := A[a, b, c, d] * B[c, f, b, g]
        @tensor ref[a, g, d, f] := hostA[a, b, c, d] * hostB[c, f, b, g]
        @test contracted isa NDArray
        @test Array(contracted) == ref

        hostX = reshape(
            ComplexF64.(1:6) .+ im .* ComplexF64.(6:-1:1), 2, 3
        )
        hostY = reshape(
            ComplexF64.(7:26) .- im .* ComplexF64.(1:20), 4, 5
        )
        X = NDArray(hostX)
        Y = NDArray(hostY)
        @tensor outer[a, c, b, d] := conj(X[a, b]) * conj(Y[c, d])
        @test Array(outer) == reshape(conj(hostX), 2, 1, 3, 1) .*
                              reshape(conj(hostY), 1, 4, 1, 5)
    end

    @testset "@tensor block and tensor network" begin
        hostA = reshape(
            collect(Float64, 1:(2 * 3 * 4 * 5 * 4 * 6)), 2, 3, 4, 5, 4, 6
        )
        hostB = reshape(collect(Float64, 1:(6 * 7 * 3)), 6, 7, 3)
        hostC = reshape(collect(Float64, 1:(5 * 2 * 7)), 5, 2, 7)
        A = NDArray(hostA)
        B = NDArray(hostB)
        C = NDArray(hostC)
        D = cuNumeric.zeros(Float64, 2, 7, 5)
        α = 0.25

        @tensor begin
            D[a, b, c] = A[a, e, f, c, f, g] * B[g, b, e] + α * C[c, a, b]
            E[a, b, c] := A[a, e, f, c, f, g] * B[g, b, e] + α * C[c, a, b]
        end
        @tensor ref[a, b, c] := hostA[a, e, f, c, f, g] * hostB[g, b, e] +
                                α * hostC[c, a, b]
        @test D isa NDArray
        @test E isa NDArray
        @test Array(D) == ref
        @test Array(E) == ref

        hostA1 = reshape(collect(Float64, 1:24), 2, 3, 4)
        hostA2 = reshape(collect(Float64, 1:120), 4, 5, 6)
        hostL = reshape(collect(Float64, 1:4), 2, 2)
        hostR = reshape(collect(Float64, 1:36), 6, 6)
        hostH = reshape(collect(Float64, 1:225), 3, 5, 3, 5)
        A1, A2 = NDArray(hostA1), NDArray(hostA2)
        L, R, H = NDArray(hostL), NDArray(hostR), NDArray(hostH)

        @tensor network[a, s1, s2, c] :=
            L[a, ap] * A1[ap, t1, b] *
            A2[b, t2, cp] * R[cp, c] *
            H[s1, s2, t1, t2]
        @tensor network_ref[a, s1, s2, c] :=
            hostL[a, ap] * hostA1[ap, t1, b] *
            hostA2[b, t2, cp] * hostR[cp, c] *
            hostH[s1, s2, t1, t2]
        @test network isa NDArray
        @test Array(network) == network_ref
    end

    @testset "tensoradd!" begin
        host = ComplexF64.(reshape(1:6, 2, 3)) .+ im .* reshape(7:12, 2, 3)
        A = NDArray(host)
        C = NDArray(fill(ComplexF64(2 - im), 3, 2))

        TO.tensoradd!(C, A, ((2, 1), ()), true, 2, 3)
        @test Array(C) ≈ 3 .* fill(ComplexF64(2 - im), 3, 2) .+
                         2 .* permutedims(conj(host))

        @tensor allocated[j, i] := conj(A[i, j])
        @test allocated isa NDArray
        @test Array(allocated) == permutedims(conj(host))

        overwrite = NDArray(fill(ComplexF64(NaN), 3, 2))
        TO.tensoradd!(overwrite, A, ((2, 1), ()), false, 1, 0)
        @test Array(overwrite) == permutedims(host)
    end

    @testset "tensortrace!" begin
        host = reshape(collect(Float64, 1:(2 * 3 * 3)), 2, 3, 3)
        A = NDArray(host)
        @tensor traced[i] := A[i, j, j]
        ref = [sum(host[i, j, j] for j in axes(host, 2)) for i in axes(host, 1)]
        @test traced isa NDArray
        @test Array(traced) == ref

        host5 = reshape(collect(Float64, 1:(2 * 3 * 3 * 4 * 4)), 2, 3, 3, 4, 4)
        A5 = NDArray(host5)
        @tensor traced2[i] := A5[i, j, j, k, k]
        ref2 = [
            sum(host5[i, j, j, k, k] for j in axes(host5, 2), k in axes(host5, 4))
            for i in axes(host5, 1)
        ]
        @test Array(traced2) == ref2

        matrix = reshape(ComplexF64.(1:9) .+ im .* ComplexF64.(9:-1:1), 3, 3)
        M = NDArray(matrix)
        @tensor scalar_trace = M[i, i]
        @test scalar_trace isa ComplexF64
        @test scalar_trace == sum(matrix[i, i] for i in axes(matrix, 1))

        conjugated_trace = TO.tensortrace(M, ((), ()), ((1,), (2,)), true)
        @test TO.tensorscalar(conjugated_trace) ==
            sum(conj(matrix[i, i]) for i in axes(matrix, 1))
        TO.tensorfree!(conjugated_trace)
    end

    @testset "tensorcontract!" begin
        hostA = ComplexF64.(reshape(1:12, 3, 4)) .+ im .* reshape(13:24, 3, 4)
        hostB = ComplexF64.(reshape(1:20, 4, 5)) .- im .* reshape(21:40, 4, 5)
        A = NDArray(hostA)
        B = NDArray(hostB)

        @tensor C[i, j] := conj(A[i, k]) * B[k, j]
        @test C isa NDArray
        @test Array(C) ≈ conj(hostA) * hostB

        seed = fill(ComplexF64(1 + 2im), 3, 5)
        accumulated = NDArray(copy(seed))
        TO.tensorcontract!(
            accumulated,
            A,
            ((1,), (2,)),
            true,
            B,
            ((1,), (2,)),
            false,
            ((1, 2), ()),
            2,
            3,
        )
        @test Array(accumulated) ≈ 3 .* seed .+ 2 .* (conj(hostA) * hostB)

        uhost = ComplexF64.(1:4) .+ im .* ComplexF64.(4:-1:1)
        vhost = ComplexF64.(5:8) .- im .* ComplexF64.(1:4)
        u = NDArray(uhost)
        v = NDArray(vhost)
        @tensor scalar_product = u[k] * v[k]
        @test scalar_product isa ComplexF64
        @test scalar_product ≈ sum(uhost .* vhost)
    end

    @testset "temporary destruction" begin
        hostA = reshape(collect(Float64, 1:12), 3, 4)
        hostB = reshape(collect(Float64, 1:20), 4, 5)
        hostD = reshape(collect(Float64, 1:10), 5, 2)
        A = NDArray(hostA)
        B = NDArray(hostB)
        D = NDArray(hostD)
        GC.gc()
        cuNumeric.drain_pending_frees!()
        current_bytes =
            cuNumeric.HAS_CUDA ?
            cuNumeric.current_device_bytes :
            cuNumeric.current_host_bytes
        baseline = current_bytes[]

        @tensor C[i, l] := A[i, j] * B[j, k] * D[k, l]
        @test current_bytes[] == baseline + C.nbytes
        @test Array(C) ≈ hostA * hostB * hostD

        TO.tensorfree!(C)
        @test C.ptr == C_NULL
        @test current_bytes[] == baseline
    end
end
