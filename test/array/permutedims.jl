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

@testset "permutedims" begin
    @testset for T in (Float32, Float64)
        A = my_rand(T, 3, 4)
        nda = NDArray(A)
        allowscalar() do
            @test safe_compare(permutedims(A), cuNumeric.transpose(nda), atol(T), rtol(T))
            @test safe_compare(permutedims(A, (2, 1)), permutedims(nda, (2, 1)), atol(T), rtol(T))
            @test safe_compare(permutedims(A), permutedims(nda), atol(T), rtol(T))
        end

        B = my_rand(T, 2, 3, 4)
        ndb = NDArray(B)
        allowscalar() do
            @test safe_compare(
                permutedims(B, (3, 1, 2)), permutedims(ndb, (3, 1, 2)), atol(T), rtol(T)
            )
            @test safe_compare(
                permutedims(B, (1, 2, 3)), permutedims(ndb, (1, 2, 3)), atol(T), rtol(T)
            )
        end
    end
    @test_throws ArgumentError permutedims(cuNumeric.ones(2, 3), (1,))
    @test_throws ArgumentError permutedims(cuNumeric.ones(2, 3), (1, 1))
    @test_throws ArgumentError permutedims(cuNumeric.ones(2, 3), (1, 3))
end

@testset "squeeze / dropdims" begin
    @testset for T in (Float32, Float64)
        A = my_rand(T, 2, 3)
        nda = NDArray(reshape(A, 2, 1, 3))
        out = squeeze(nda)
        @test size(out) == (2, 3)
        allowscalar() do
            @test safe_compare(A, out, atol(T), rtol(T))
        end
        @test size(squeeze(NDArray(A))) == (2, 3)

        dropped = dropdims(nda; dims=2)
        @test size(dropped) == (2, 3)
        allowscalar() do
            @test safe_compare(A, dropped, atol(T), rtol(T))
        end
        @test size(squeeze(nda, 2)) == (2, 3)
        @test_throws DimensionMismatch squeeze(nda, 1)
        @test_throws ArgumentError dropdims(nda; dims=4)
    end
end
