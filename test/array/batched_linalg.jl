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

# Batched (3D+) linear algebra. These entry points are deliberately not
# LinearAlgebra methods: Base has no batched equivalent.

const N_BATCH = 3

function batched_spd(::Type{T}, b, n) where {T}
    RT = real(T)
    out = zeros(T, b, n, n)
    for i in 1:b
        B = my_rand(T, n, n; L=RT(-1), R=RT(1))
        out[i, :, :] = B * B' + T(n) * Matrix{T}(I, n, n)
    end
    return out
end

function batched_random(::Type{T}, dims...) where {T}
    RT = real(T)
    return my_rand(T, dims...; L=RT(-1), R=RT(1))
end

@testset "batched_solve" begin
    @testset verbose=true for T in Base.uniontypes(cuNumeric.SUPPORTED_SOLVE_TYPES)
        n, nrhs = 4, 2
        A_ref = batched_spd(T, N_BATCH, n)
        b_ref = batched_random(T, N_BATCH, n, nrhs)

        x = cuNumeric.batched_solve(cuNumeric.NDArray(A_ref), cuNumeric.NDArray(b_ref))

        allowscalar() do
            X = Array(x)
            @test size(X) == (N_BATCH, n, nrhs)
            for i in 1:N_BATCH
                @test isapprox(
                    A_ref[i, :, :] \ b_ref[i, :, :], X[i, :, :]; atol=atol(T), rtol=rtol(T)
                )
            end
        end
    end
end

@testset "batched_solve promotion" begin
    @testset verbose=true for T in (Int32, Int64, Bool)
        A = cuNumeric.NDArray(reshape(T[1, 0, 0, 1], 1, 2, 2))
        b = cuNumeric.NDArray(reshape(T[1, 1], 1, 2, 1))

        @test_throws "Implicit promotion" cuNumeric.batched_solve(A, b)

        allowpromotion() do
            x = cuNumeric.batched_solve(A, b)
            allowscalar() do
                @test safe_compare(
                    reshape(Float64[1, 1], 1, 2, 1), x, atol(Float64), rtol(Float64)
                )
            end
        end
    end
end

@testset "batched_solve rejects 2D input" begin
    A = cuNumeric.zeros(Float64, 3, 3)
    b = cuNumeric.zeros(Float64, 3, 1)
    @test_throws "requires shape (b,m,m)" cuNumeric.batched_solve(A, b)
end

@testset "batched_cholesky" begin
    @testset verbose=true for T in Base.uniontypes(cuNumeric.SUPPORTED_CHOLESKY_TYPES)
        n = 4
        A_ref = batched_spd(T, N_BATCH, n)
        out = cuNumeric.batched_cholesky(cuNumeric.NDArray(A_ref))

        allowscalar() do
            L = Array(out)
            @test size(L) == (N_BATCH, n, n)
            for i in 1:N_BATCH
                Li = L[i, :, :]
                @test istril(Li)
                @test isapprox(A_ref[i, :, :], Li * Li'; atol=atol(T), rtol=rtol(T))
            end
        end
    end
end

@testset "batched_cholesky promotion" begin
    @testset verbose=true for T in (Int32, Int64, Bool)
        vals = reshape(T[1, 0, 0, 1], 1, 2, 2)
        A = cuNumeric.NDArray(vals)
        @test_throws "Implicit promotion" cuNumeric.batched_cholesky(A)
        allowpromotion() do
            out = cuNumeric.batched_cholesky(A)
            allowscalar() do
                @test safe_compare(Float64.(vals), out, atol(Float64), rtol(Float64))
            end
        end
    end
end

function batched_eigen_residual(A_ref, values, vectors, i)
    C = eltype(values)
    Ai = C.(A_ref[i, :, :])
    Vi = vectors[i, :, :]
    return maximum(abs.(Ai * Vi .- Vi * Diagonal(values[i, :])))
end

@testset "batched_eigen" begin
    @testset verbose=true for T in Base.uniontypes(cuNumeric.SUPPORTED_EIG_TYPES)
        n = 4
        A_ref = batched_random(T, N_BATCH, n, n)
        values, vectors = cuNumeric.batched_eigen(cuNumeric.NDArray(A_ref))

        allowscalar() do
            vals, vecs = Array(values), Array(vectors)
            @test size(vals) == (N_BATCH, n)
            @test size(vecs) == (N_BATCH, n, n)
            @test eltype(vals) == complex(T)
            for i in 1:N_BATCH
                @test batched_eigen_residual(A_ref, vals, vecs, i) <=
                    max(atol(T), rtol(T) * n)
                @test isapprox(
                    sort(vals[i, :]; by=x -> (real(x), imag(x))),
                    sort(LinearAlgebra.eigvals(A_ref[i, :, :]); by=x -> (real(x), imag(x))),
                    atol=atol(T),
                    rtol=rtol(T),
                )
            end
        end
    end
end

@testset "batched_eigvals" begin
    @testset verbose=true for T in Base.uniontypes(cuNumeric.SUPPORTED_EIG_TYPES)
        n = 4
        A_ref = batched_random(T, N_BATCH, n, n)
        values = cuNumeric.batched_eigvals(cuNumeric.NDArray(A_ref))

        allowscalar() do
            vals = Array(values)
            @test size(vals) == (N_BATCH, n)
            for i in 1:N_BATCH
                @test isapprox(
                    sort(vals[i, :]; by=x -> (real(x), imag(x))),
                    sort(LinearAlgebra.eigvals(A_ref[i, :, :]); by=x -> (real(x), imag(x))),
                    atol=atol(T),
                    rtol=rtol(T),
                )
            end
        end
    end
end

@testset "batched dimension limits" begin
    # Exactly one batch dimension: 2D belongs to the LinearAlgebra entry points,
    # and 4D exceeds both the POTRF task and Legate's launch-domain construction.
    @testset "$f" for f in
                      (cuNumeric.batched_cholesky, cuNumeric.batched_eigen,
        cuNumeric.batched_eigvals)
        @test_throws ArgumentError f(cuNumeric.zeros(Float64, 3, 3))
        @test_throws ArgumentError f(cuNumeric.zeros(Float64, 2, 2, 3, 3))
        @test_throws ArgumentError f(cuNumeric.zeros(Float64, 2, 3, 4))
    end

    @test_throws ArgumentError cuNumeric.batched_solve(
        cuNumeric.zeros(Float64, 2, 2, 3, 3), cuNumeric.zeros(Float64, 2, 2, 3, 1)
    )
end
