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

rtol(::Type{Float16}) = 1e-2
rtol(::Type{Float32}) = 1e-5
rtol(::Type{Float64}) = 1e-12
rtol(::Type{I}) where {I<:Integer} = rtol(float(I))
atol(::Type{Float16}) = 1e-3
atol(::Type{Float32}) = 1e-8
atol(::Type{Float64}) = 1e-15
atol(::Type{I}) where {I<:Integer} = atol(float(I))
rtol(::Type{Complex{T}}) where {T} = rtol(T)
atol(::Type{Complex{T}}) where {T} = atol(T)

include("tests/daxpy.jl")
include("tests/daxpy_advanced.jl")
include("tests/elementwise.jl")
include("tests/slicing.jl")
include("tests/sgemm.jl")
# include("tests/custom_cuda.jl")

@testset verbose = true "AXPY" begin
    @testset axpy_basic()
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

    function test_acosh(N, T)
        julia_arr2 = rand(T, N)
        cunumeric_arr2 = cuNumeric.zeros(T, N)
        julia_arr2[julia_arr2 .< 1.0] = 1.0
        @allowscalar for i in 1:N
            cunumeric_arr2[i] = julia_arr2[i]
        end
        cunumeric_res = func.(cunumeric_arr2)
        cunumeric_res2 = map(func, cunumeric_arr)
        julia_res2 = func.(julia_arr)
        allowscalar() do
            @test cuNumeric.compare(julia_res, cunumeric_res, atol(T), rtol(T))
            @test cuNumeric.compare(julia_res, cunumeric_res2, atol(T), rtol(T))
        end
    end

    function test_conj()
        #! once we support compelx numbers do this
    end

    @testset for T in Base.uniontypes(cuNumeric.SUPPORTED_TYPES)

        # Make input arrays we can re-use
        julia_arr = rand(T, N)
        cunumeric_arr = cuNumeric.zeros(T, N)
        @allowscalar for i in 1:N
            cunumeric_arr[i] = julia_arr[i]
        end

        @testset for func in keys(cuNumeric.floaty_unary_ops_no_args)
            # Julia throws error cause domain is 
            # restricted to > 1
            func == Base.acosh && test_acosh(N, T)

            cunumeric_res = func.(cunumeric_arr)
            cunumeric_res2 = map(func, cunumeric_arr)
            julia_res = func.(julia_arr)
            allowscalar() do
                @test cuNumeric.compare(julia_res, cunumeric_res, atol(T), rtol(T))
                @test cuNumeric.compare(julia_res, cunumeric_res2, atol(T), rtol(T))
            end
        end

        #TODO more generic tests for unary_op_map_no_args
    end
end

@testset verbose = true "Unary Reductions" begin
    N = 100

    @testset for T in Base.uniontypes(cuNumeric.SUPPORTED_TYPES)
        julia_arr = rand(T, N)
        cunumeric_arr = cuNumeric.zeros(T, N)
        @allowscalar for i in 1:N
            cunumeric_arr[i] = julia_arr[i]
        end

        @testset for reduction in keys(cuNumeric.unary_reduction_map)
            cunumeric_res = reduction(cunumeric_arr)
            julia_res = reduction(julia_arr)
            allowscalar() do
                @test cuNumeric.compare([julia_res], cunumeric_res, atol(T), rtol(T))
            end
        end
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
    @allowscalar for i in 1:N
        cunumeric_arr1[i] = julia_arr1[i]
        cunumeric_arr2[i] = julia_arr2[i]
    end

    ## GENERATE TEST ON RANDOM FLOAT64s FOR EACH UNARY OP
    @testset for func in keys(cuNumeric.binary_op_map)
        cunumeric_res = func(cunumeric_arr1, cunumeric_arr2)
        cunumeric_res2 = map(func, cunumeric_arr1, cunumeric_arr2)
        julia_res .= func.(julia_arr1, julia_arr2)
        allowscalar() do
            @test cuNumeric.compare(julia_res, cunumeric_res, max_diff)
            @test cuNumeric.compare(julia_res, cunumeric_res2, max_diff)
        end
    end

    @testset "Type and Shape Promotion" begin
        cunumeric_arr3 = cuNumeric.zeros(Float32, N)
        cunumeric_int64 = cuNumeric.zeros(Int64, N)
        cunumeric_int32 = cuNumeric.zeros(Int32, N)
        cunumeric_arr5 = cuNumeric.zeros(Float64, N, N)


        @test_throws "Detected promotion" cunumeric_arr3 + cunumeric_arr1
        @test_throws "Detected promotion" map(+, cunumeric_arr3, cunumeric_arr1)
        @test_throws DimensionMismatch cunumeric_arr1 + cunumeric_arr5
        @test_throws DimensionMismatch cunumeric_arr1 / cunumeric_arr5

        @test cunumeric_arr1 == cunumeric_int64 + cunumeric_arr1

    end

    @testset "Copy-To" begin
        a = cuNumeric.zeros(2, 2)
        b = cuNumeric.ones(2, 2)
        copyto!(a, b);
        @test a == b
    end
end

@testset verbose = true "Slicing Tests" begin
    max_diff = Float64(1e-4)
    @testset slicing(max_diff)
end

# @testset verbose = true "CUDA Tests" begin
#     max_diff = Float32(1e-4)
#     @testset binaryop(max_diff)
# end
