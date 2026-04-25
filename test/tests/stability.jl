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

@testset verbose = true "core" begin
    a = cuNumeric.zeros(5)
    b = cuNumeric.zeros(Float64, 3, 4)
    @test @inferred(size(a)) !== nothing
    @test @inferred(size(b)) !== nothing
    @test @inferred(cuNumeric.shape(a)) !== nothing
    @test @inferred(cuNumeric.shape(b)) !== nothing
end

@testset verbose = true "construction" begin
    # zeros, zeros_like, ones, rand, fill, trues, falses\
    for constructor in (:zeros, :ones)
        @eval begin
            @test @inferred(cuNumeric.$(constructor)(Float64, 3, 2)) !== nothing
            @test @inferred(cuNumeric.$(constructor)(Float64, (3, 4))) !== nothing
            @test @inferred(cuNumeric.$(constructor)(3, 5, 6)) !== nothing
            @test @inferred(cuNumeric.$(constructor)((3,))) !== nothing
            @test @inferred(cuNumeric.$(constructor)()) !== nothing
            @test @inferred(cuNumeric.$(constructor)(Int64)) !== nothing
        end
    end
    a = cuNumeric.zeros(Float64, 5, 3)
    @test @inferred(cuNumeric.zeros_like(a)) !== nothing

    for constructor in (:trues, :falses)
        @eval begin
            @test @inferred(cuNumeric.$(constructor)(5)) !== nothing
            @test @inferred(cuNumeric.$(constructor)((5, 4))) !== nothing
            @test @inferred(cuNumeric.$(constructor)(3, 4, 5)) !== nothing
        end
    end

    @test @inferred(cuNumeric.fill(2.0, 3, 4)) !== nothing
    @test @inferred(cuNumeric.fill(2, (3, 4))) !== nothing
    @test @inferred(cuNumeric.fill(2.0, 3)) !== nothing

    @test @inferred(cuNumeric.rand(4, 3)) !== nothing
    @test @inferred(cuNumeric.rand(Float32, 5)) !== nothing

    # NDArray from Julia Array (Parent-stable attachment)
    @test @inferred(cuNumeric.NDArray(rand(10))) !== nothing
    @test @inferred(cuNumeric.NDArray(rand(Float32, 3, 3))) !== nothing
end

@testset verbose = true "conversion" begin
    # cast to array, as_type
    a = cuNumeric.zeros(Float64, 5, 5)
    @test @inferred(Array(a)) !== nothing
    @test @inferred(Array{Float32}(a)) !== nothing
    @test @inferred(cuNumeric.as_type(a, Float32)) !== nothing
    @test @inferred(cuNumeric.as_type(a, Int64)) !== nothing
end

@testset verbose = true "indexing" begin
    # getindex, setindex!, copy, copyto!, fill!, as_type
    a = cuNumeric.zeros(Float32, 5, 5)
    b = cuNumeric.zeros(Int32, 11)

    @test @inferred(a[1:3, 1:3]) !== nothing
    @test @inferred(a[2, 1:3]) !== nothing
    @test @inferred(a[1, 1:3] .+ b[1:3]) !== nothing
    @test @inferred(b[1:5]) !== nothing
    # @test @inferred(a[1:3, 1:end]) !== nothing
    allowscalar() do
        @test @inferred(a[1, 2]) !== nothing
    end
end

@testset verbose = true "broadcasting" begin
    a = cuNumeric.ones(Float32, 3, 3)
    b = cuNumeric.ones(Int32, 3, 3)
    @test @inferred(5 .* a) !== nothing
    @test @inferred(5.0f0 .* a) !== nothing
    @test @inferred(5 * a) !== nothing
    @test @inferred(5.0f0 * a) !== nothing

    @test @inferred(a .* b) !== nothing
    @test @inferred(a .+ b) !== nothing
    @test @inferred(a ./ b) !== nothing
    @test @inferred(((a .* b) .+ a) .* 2.0f0) !== nothing
end
