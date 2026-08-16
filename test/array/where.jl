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

# `where` selects elementwise and `findall` reports the true positions, so both
# are exact: compare against the host reference with `==` rather than a
# tolerance.

@testset "where elementwise select" begin
    N = 16
    @testset verbose = true for T in Base.uniontypes(cuNumeric.SUPPORTED_FLOAT_TYPES)
        x_ref = my_rand(T, N)
        y_ref = my_rand(T, N)
        x = NDArray(x_ref)
        y = NDArray(y_ref)

        mask = x .> y
        mask_ref = x_ref .> y_ref
        @test Array(mask) == mask_ref

        @test Array(cuNumeric.where(mask, x, y)) == ifelse.(mask_ref, x_ref, y_ref)
        @test Array(ifelse(mask, x, y)) == ifelse.(mask_ref, x_ref, y_ref)
    end
end

@testset "where with scalar branches" begin
    N = 16
    x_ref = my_rand(Float32, N)
    x = NDArray(x_ref)
    mask = x .> 0.0f0
    mask_ref = x_ref .> 0.0f0

    @test Array(cuNumeric.where(mask, x, 0.0f0)) == ifelse.(mask_ref, x_ref, 0.0f0)
    @test Array(cuNumeric.where(mask, 0.0f0, x)) == ifelse.(mask_ref, 0.0f0, x_ref)
    @test Array(cuNumeric.where(mask, 1.0f0, -1.0f0)) == ifelse.(mask_ref, 1.0f0, -1.0f0)
    @test Array(ifelse(mask, x, 0.0f0)) == ifelse.(mask_ref, x_ref, 0.0f0)
end

@testset "where broadcasts operand shapes" begin
    N = 8
    x_ref = my_rand(Float32, N, N)
    x = NDArray(x_ref)
    mask = x .> 0.0f0
    mask_ref = x_ref .> 0.0f0

    # A scalar branch is a 0-d operand broadcast against the 2D condition.
    out = cuNumeric.where(mask, x, 0.0f0)
    @test size(out) == (N, N)
    @test Array(out) == ifelse.(mask_ref, x_ref, 0.0f0)
end

@testset "where promotes branches" begin
    N = 8
    x = NDArray(my_rand(Float32, N))
    y = NDArray(my_rand(Float64, N))
    mask = x .> 0.0f0

    @test_throws "Implicit promotion" cuNumeric.where(mask, x, y)

    allowpromotion() do
        out = cuNumeric.where(mask, x, y)
        @test eltype(out) == Float64
        @test Array(out) == ifelse.(Array(mask), Float64.(Array(x)), Array(y))
    end
end

@testset "findall" begin
    for ref in ([false, true, false, true, true], falses(6), trues(4))
        cond = NDArray(collect(ref))
        idx = findall(cond)
        @test idx isa NDArray{Int64,1}
        @test Array(idx) == findall(ref)
    end
end

@testset "findall rejects higher dimensions" begin
    cond = NDArray(Bool[true false; false true])
    @test_throws "only supported for 1D" findall(cond)
end
