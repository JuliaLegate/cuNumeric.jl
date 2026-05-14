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
=#

#= Broadcast fusion arg buffer tests.
 * Tests every combination of array and scalar argument positions
 * to verify the arg_map protocol correctly reconstructs the CUDA
 * kernel arg buffer.
 *
 * Pattern naming convention:
 *   A = NDArray input
 *   S = scalar input
 *   Position = left-to-right in the broadcast expression
 *   "same" = same array appears multiple times (deduplication test)
=#

function test_broadcast_fusion(; T=Float32, N=100, atol=1e-5, rtol=1e-5)
    # Create test arrays with known non-zero values
    julia_a = rand(T, N)
    julia_b = rand(T, N)
    julia_c = rand(T, N)

    a = @allowscalar NDArray(julia_a)
    b = @allowscalar NDArray(julia_b)
    c = @allowscalar NDArray(julia_c)

    s1 = T(2.5)
    s2 = T(1.0)
    s3 = T(0.5)

    # two different arrays
    @testset "A + B (two different arrays)" begin
        expected = julia_a .+ julia_b
        result = a .+ b
        @allowscalar @test cuNumeric.compare(expected, result, atol, rtol)
    end

    # same array, deduplication
    @testset "A + A (same array twice)" begin
        expected = julia_a .+ julia_a
        result = a .+ a
        @allowscalar @test cuNumeric.compare(expected, result, atol, rtol)
    end

    # array then scalar
    @testset "A + scalar (array first)" begin
        expected = julia_a .+ s1
        result = a .+ s1
        @allowscalar @test cuNumeric.compare(expected, result, atol, rtol)
    end

    # scalar then array
    @testset "scalar + A (scalar first)" begin
        expected = s1 .+ julia_a
        result = s1 .+ a
        @allowscalar @test cuNumeric.compare(expected, result, atol, rtol)
    end

    # array, two scalars, fused
    @testset "A * scalar - scalar (fused)" begin
        expected = julia_a .* s1 .- s2
        result = a .* s1 .- s2
        @allowscalar @test cuNumeric.compare(expected, result, atol, rtol)
    end

    # scalar-array-scalar
    @testset "scalar * A + scalar" begin
        expected = s1 .* julia_a .+ s2
        result = s1 .* a .+ s2
        @allowscalar @test cuNumeric.compare(expected, result, atol, rtol)
    end

    # two arrays then scalar
    @testset "A + B + scalar" begin
        expected = julia_a .+ julia_b .+ s1
        result = a .+ b .+ s1
        @allowscalar @test cuNumeric.compare(expected, result, atol, rtol)
    end

    # scalar then two arrays
    @testset "scalar + A + B" begin
        expected = s1 .+ julia_a .+ julia_b
        result = s1 .+ a .+ b
        @allowscalar @test cuNumeric.compare(expected, result, atol, rtol)
    end

    # same array twice + scalar (dedup)
    @testset "A + A + scalar (dedup + scalar)" begin
        expected = julia_a .+ julia_a .+ s1
        result = a .+ a .+ s1
        @allowscalar @test cuNumeric.compare(expected, result, atol, rtol)
    end

    # three different arrays
    @testset "A + B + C (three arrays)" begin
        expected = julia_a .+ julia_b .+ julia_c
        result = a .+ b .+ c
        @allowscalar @test cuNumeric.compare(expected, result, atol, rtol)
    end

    # same array three times (triple dedup)
    @testset "A + A + A (triple dedup)" begin
        expected = julia_a .+ julia_a .+ julia_a
        result = a .+ a .+ a
        @allowscalar @test cuNumeric.compare(expected, result, atol, rtol)
    end

    # two scalars then array
    @testset "scalar * scalar + A" begin
        expected = s1 .* s2 .+ julia_a
        result = s1 .* s2 .+ a
        @allowscalar @test cuNumeric.compare(expected, result, atol, rtol)
    end

    # multiply, two arrays (different PTX kernel name collision test)
    @testset "A * B (multiply)" begin
        expected = julia_a .* julia_b
        result = a .* b
        @allowscalar @test cuNumeric.compare(expected, result, atol, rtol)
    end

    # same array squared, dedup
    @testset "A * A (self multiply, dedup)" begin
        expected = julia_a .* julia_a
        result = a .* a
        @allowscalar @test cuNumeric.compare(expected, result, atol, rtol)
    end

    # subtraction, order matters
    @testset "A - B (subtraction)" begin
        expected = julia_a .- julia_b
        result = a .- b
        @allowscalar @test cuNumeric.compare(expected, result, atol, rtol)
    end

    # scalar minus array
    @testset "scalar - A" begin
        expected = s1 .- julia_a
        result = s1 .- a
        @allowscalar @test cuNumeric.compare(expected, result, atol, rtol)
    end

    # three arrays, mixed ops
    @testset "A * B + C (three arrays, mixed ops)" begin
        expected = julia_a .* julia_b .+ julia_c
        result = a .* b .+ c
        @allowscalar @test cuNumeric.compare(expected, result, atol, rtol)
    end

    # two arrays fused, then scaled
    @testset "(A + B) * scalar" begin
        expected = (julia_a .+ julia_b) .* s1
        result = (a .+ b) .* s1
        @allowscalar @test cuNumeric.compare(expected, result, atol, rtol)
    end

    # scalar-array pairs
    @testset "scalar * A + scalar * B" begin
        expected = s1 .* julia_a .+ s2 .* julia_b
        result = s1 .* a .+ s2 .* b
        @allowscalar @test cuNumeric.compare(expected, result, atol, rtol)
    end

    # same array subtracted, should be all zeros
    @testset "A - A (same array, expect zeros)" begin
        expected = julia_a .- julia_a
        result = a .- a
        @allowscalar @test cuNumeric.compare(expected, result, atol, rtol)
    end
end
