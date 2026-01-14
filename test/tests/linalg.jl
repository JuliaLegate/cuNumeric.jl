#= Copyright 2025 Northwestern University, 
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
 *            Nader Rahal <naderrahhal2026@u.northwestern.edu>
=#

@testset "transpose" begin
    A = rand(Float64, 4, 3)
    nda = cuNumeric.NDArray(A)

    ref = transpose(A)
    out = cuNumeric.transpose(nda)

    allowscalar() do
        @test cuNumeric.compare(ref, out, atol(Float64), rtol(Float64))
    end
end

@testset "eye" begin
    for T in (Float32, Float64, Int32)
        n = 5
        ref = Matrix{T}(I, n, n)
        out = cuNumeric.eye(n; T=T)
        allowscalar() do
            @test cuNumeric.compare(ref, out, atol(T), rtol(T))
        end
    end
end

@testset "trace" begin
    A = rand(Float64, 6, 6)
    nda = cuNumeric.NDArray(A)

    ref = tr(A)
    out = cuNumeric.trace(nda)

    allowscalar() do
        @test ref ≈ out[1] atol=atol(Float32) rtol=rtol(Float32)
    end
end

@testset "trace with offset" begin
    A = rand(Float32, 5, 5)
    nda = cuNumeric.NDArray(A)

    for k in (-2, -1, 0, 1, 2)
        ref = sum(diag(A, k))
        out = cuNumeric.trace(nda; offset=k)

        allowscalar() do
            @test ref ≈ out[1] atol=atol(Float32) rtol=rtol(Float32)
        end
    end
end

@testset "diag" begin
    A = rand(Int, 6, 6)
    nda = cuNumeric.NDArray(A)

    for k in (-2, 0, 3)
        ref = diag(A, k)
        out = cuNumeric.diag(nda; k=k)

        allowscalar() do
            @test cuNumeric.compare(ref, out, atol(Int32), rtol(Int32))
        end
    end
end

# @testset "ravel" begin
#     A = reshape(collect(1:12), 3, 4)
#     nda = cuNumeric.NDArray(A)

#     ref = vec(A)
#     out = cuNumeric.ravel(nda)

#     allowscalar() do
#         @test cuNumeric.compare(ref, out, atol(Int32), rtol(Int32))
#     end
# end

@testset "unique" begin
    A = [1, 2, 2, 3, 4, 4, 4, 5]
    nda = cuNumeric.NDArray(A)

    ref = unique(A)
    out = cuNumeric.unique(nda)

    @test sort(Array(out)) == sort(ref)
end
