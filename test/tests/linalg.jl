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

@testset "solve diagonal" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64, Int8, Int16, Int32, Int64)
        n = 4
        T_comp = T <: Integer ? Float64 : T
        A = cuNumeric.zeros(T, n, n)
        b = cuNumeric.zeros(T, n, 1)
        cuNumeric.@allowscalar for i in 1:n
            A[i, i] = T(4)
            b[i, 1] = T(1)
        end
        x = cuNumeric.solve(cuNumeric.as_type(A, T_comp), cuNumeric.as_type(b, T_comp))
        allowscalar() do
            @test cuNumeric.compare(fill(T_comp(0.25), n, 1), x, atol(T_comp), rtol(T_comp))
        end
    end
end

@testset "solve identity" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64, Int8, Int16, Int32, Int64)
        n = 4
        T_comp = T <: Integer ? Float64 : T
        A = cuNumeric.NDArray(Matrix{T}(I, n, n))
        b = cuNumeric.NDArray(reshape(T.(collect(1:n)), n, 1))
        x = cuNumeric.solve(cuNumeric.as_type(A, T_comp), cuNumeric.as_type(b, T_comp))
        ref = reshape(Float64.(collect(1:n)), n, 1)
        allowscalar() do
            @test cuNumeric.compare(ref, x, atol(T_comp), rtol(T_comp))
        end
    end
end

@testset "solve general" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64, Int8, Int16, Int32, Int64)
        T_comp = T <: Integer ? Float64 : T
        A_ref = T[2 1; 5 7]
        b_ref = T[11; 13;;]
        A = cuNumeric.NDArray(A_ref)
        b = cuNumeric.NDArray(b_ref)
        x = cuNumeric.solve(cuNumeric.as_type(A, T_comp), cuNumeric.as_type(b, T_comp))
        ref = Float64.(A_ref) \ Float64.(b_ref)
        allowscalar() do
            @test cuNumeric.compare(ref, x, atol(T_comp), rtol(T_comp))
        end
    end
end