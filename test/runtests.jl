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
=#

using Test
using cuNumeric
using LinearAlgebra

include("tests/daxpy.jl")
include("tests/daxpy_advanced.jl")
include("tests/elementwise.jl")
include("tests/slicing.jl")
include("tests/sgemm.jl")

@testset verbose = true "DAXPY" begin
    @testset daxpy_basic()
    @testset daxpy_advanced()
end

@testset verbose = true "Operators" begin
    @testset elementwise()
end

@testset verbose = true "SGEMM" begin
    max_diff = Float32(1e-4)
    @warn "SGEMM has some precision issues, using tol $(max_diff) 🥲"
    @testset sgemm(max_diff)
end

#*TODO ADD IN PLACE VARIANTS
#*TODO TEST VARIANT OVER DIMS
@testset verbose = true "Unary Ops w/o Args" begin
    N = 100
    max_diff = 1e-13

    # Make input arrays we can re-use
    julia_arr = rand(Float64, N)
    julia_res = zeros(Float64, size(julia_arr))
    cunumeric_arr = cuNumeric.zeros(Float64, N)
    for i in 1:N
        cunumeric_arr[i] = julia_arr[i]
    end

    ## GENERATE TEST ON RANDOM FLOAT64s FOR EACH UNARY OP
    @testset for func in keys(cuNumeric.unary_op_map_no_args)

        # TODO Custom functions don't have a Julia equivalent
        # so we cannot test them.
        if func isa Symbol
            continue
        end

        cunumeric_res = func(cunumeric_arr)
        cunumeric_res2 = map(func, cunumeric_arr)
        julia_res .= func.(julia_arr)
        @test cuNumeric.compare(julia_res, cunumeric_res, max_diff)
        @test cuNumeric.compare(julia_res, cunumeric_res2, max_diff)
    end
end

@testset verbose = true "Unary Reductions" begin
    N = 100
    max_diff = 1e-13

    # Make input arrays we can re-use
    julia_arr = rand(Float64, N)
    cunumeric_arr = cuNumeric.zeros(Float64, N)
    for i in 1:N
        cunumeric_arr[i] = julia_arr[i]
    end

    ## GENERATE TEST ON RANDOM FLOAT64s FOR EACH UNARY OP
    @testset for reduction in keys(cuNumeric.unary_reduction_map)
        cunumeric_res = reduction(cunumeric_arr)
        julia_res = reduction(julia_arr)
        @test cuNumeric.compare([julia_res], cunumeric_res, max_diff)
    end
end

#*TODO ADD IN PLACE VARIANTS
@testset verbose = true "Binary Ops" begin
    N = 100
    max_diff = 1e-13

    # Make input arrays we can re-use
    julia_arr1 = rand(Float64, N)
    julia_arr2 = rand(Float64, N)
    julia_res = zeros(Float64, N)

    cunumeric_arr1 = cuNumeric.zeros(Float64, N)
    cunumeric_arr2 = cuNumeric.zeros(Float64, N)
    for i in 1:N
        cunumeric_arr1[i] = julia_arr1[i]
        cunumeric_arr2[i] = julia_arr2[i]
    end

    ## GENERATE TEST ON RANDOM FLOAT64s FOR EACH UNARY OP
    @testset for func in keys(cuNumeric.binary_op_map)
        cunumeric_res = func(cunumeric_arr1, cunumeric_arr2)
        cunumeric_res2 = map(func, cunumeric_arr1, cunumeric_arr2)
        julia_res .= func.(julia_arr1, julia_arr2)
        @test cuNumeric.compare(julia_res, cunumeric_res, max_diff)
        @test cuNumeric.compare(julia_res, cunumeric_res2, max_diff)
    end
end

@testset verbose = true "Slicing Tests" begin
    max_diff = Float64(1e-4)
    @testset slicing(max_diff)
end
