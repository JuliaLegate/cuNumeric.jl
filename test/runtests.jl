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
using LinearAlgebra
using Random
import Random: rand

const VERBOSE = get(ENV, "VERBOSE", "1") != "0"
const run_gpu_tests = get(ENV, "GPUTESTS", "1") != "0"
@info "Run GPU Tests: $(run_gpu_tests)"

if run_gpu_tests 
    using CUDA
    import CUDA: i32
    VERBOSE && println(CUDA.versioninfo())
end 

if run_gpu_tests && !CUDA.functional()
    error(
        "You asked for CUDA tests, but they are disabled because no functional CUDA device was detected."
    )
end

using cuNumeric
VERBOSE && cuNumeric.versioninfo()

include("tests/util.jl")
include("tests/axpy.jl")
include("tests/axpy_advanced.jl")
include("tests/elementwise.jl")
include("tests/slicing.jl")
include("tests/gemm.jl")
include("tests/unary_tests.jl")
include("tests/binary_tests.jl")
include("tests/scoping.jl")
include("tests/scoping-advanced.jl")

@testset verbose = true "AXPY" begin
    N = 100
    @testset verbose = true for T in Base.uniontypes(cuNumeric.SUPPORTED_FLOAT_TYPES)
        @testset "basic" axpy_basic(T, N)
        @testset "advanced" axpy_advanced(T, N)
    end
end

@testset verbose = true "Operators" begin
    @testset elementwise()
end

@testset verbose = true "GEMM" begin
    N = 50
    M = 25
    @testset verbose = true for T in Base.uniontypes(cuNumeric.SUPPORTED_TYPES)
        # @warn "SGEMM has some precision issues, using tol $(rtol(T)) 🥲"
        gemm(N, M, T, rtol(T))
    end
end

#* TODO TEST VARIANT OVER DIMS
@testset verbose = true "Unary Ops w/o Args" begin
    N = 100 # keep as perfect square

    @testset for T in Base.uniontypes(cuNumeric.SUPPORTED_TYPES)
        allowpromotion(T == Bool || T == Int32) do
            test_unary_function_set(cuNumeric.floaty_unary_ops_no_args, T, N)
        end

        allowpromotion(T == Bool) do
            test_unary_function_set(cuNumeric.unary_op_map_no_args, T, N)
        end
        # Special cases for unary ops that dont use . syntax
        @testset "- (Negation)" begin
            arr = my_rand(T, N)
            arr_cn = @allowscalar NDArray(arr)

            allowscalar() do
                allowpromotion(T == Bool) do
                    T_OUT = T == Bool ? cuNumeric.DEFAULT_INT : T
                    @test cuNumeric.compare(T_OUT.(-arr), -arr_cn, atol(T), rtol(T))
                end
            end
        end

        #!SPECIAL CASES (!, -)
    end
end

@testset verbose = true "Unary Reductions" begin
    N = 100

    @testset for T in Base.uniontypes(cuNumeric.SUPPORTED_TYPES)
        julia_arr = my_rand(T, N)
        cunumeric_arr = @allowscalar NDArray(julia_arr)

        @testset "$(reduction)" for reduction in keys(cuNumeric.unary_reduction_map)
            enable_sum_promotion = (T == Int32 || T == Bool) && (reduction == Base.sum)
            enable_prod_promotion = (T == Int32) && (reduction == Base.prod)

            # Test promotion errors cause we can:
            if enable_sum_promotion
                @test_throws "Implicit promotion" reduction(cunumeric_arr)
            end

            if enable_prod_promotion
                @test_throws "Implicit promotion" reduction(cunumeric_arr)
            end

            allowpromotion(enable_sum_promotion || enable_prod_promotion) do
                cunumeric_res = reduction(cunumeric_arr)
                julia_res = reduction(julia_arr)

                allowscalar() do
                    # assumes 0D result
                    @test isapprox(julia_res, cunumeric_res[]; atol=atol(T), rtol=rtol(T))
                end
            end
        end
    end

    # Test things that only work on Booleans
    julia_bools = rand(Bool, N)
    allowscalar() do
        cunumeric_bools = NDArray(julia_bools)
        @test any(julia_bools) == any(cunumeric_bools)[]
        @test all(julia_bools) == all(cunumeric_bools)[]
    end
end

@testset verbose = true "Binary Ops" begin
    N = 100

    @testset for T in Base.uniontypes(cuNumeric.SUPPORTED_TYPES)
        allowpromotion(T == Bool || T == Int32) do
            test_binary_function_set(cuNumeric.floaty_binary_op_map, T, N)
        end

        allowpromotion(T == Bool) do
            test_binary_function_set(cuNumeric.binary_op_map, T, N)
        end

        # Special cases
        @testset "lcm, gcd, ==, !=" begin
            arr_jl = my_rand(T, N)
            arr_jl2 = my_rand(T, N)
            arr_cn = @allowscalar NDArray(arr_jl)
            arr_cn2 = @allowscalar NDArray(arr_jl2)

            if T <: cuNumeric.SUPPORTED_INT_TYPES
                allowscalar() do
                    @test cuNumeric.compare(
                        lcm.(arr_jl, arr_jl2), lcm.(arr_cn, arr_cn2), atol(T), rtol(T)
                    )
                    @test cuNumeric.compare(
                        gcd.(arr_jl, arr_jl2), gcd.(arr_cn, arr_cn2), atol(T), rtol(T)
                    )
                end
            end

            allowscalar() do
                @test unwrap(arr_cn == arr_cn)
                @test !unwrap(arr_cn == arr_cn2)
                @test unwrap(arr_cn != arr_cn2)
                @test !unwrap(arr_cn != arr_cn)
                @test unwrap(all(arr_cn .== arr_cn))
            end
        end
    end

    @testset "Type and Shape Promotion" begin
        cunumeric_arr1 = cuNumeric.zeros(Float64, N)
        cunumeric_arr3 = cuNumeric.zeros(Float32, N)
        cunumeric_int64 = cuNumeric.zeros(Int64, N)
        cunumeric_int32 = cuNumeric.zeros(Int32, N)
        cunumeric_arr5 = cuNumeric.zeros(Float64, N - 1, N - 1)
        @test_throws "Implicit promotion" cunumeric_arr3 .+ cunumeric_arr1
        @test_throws "Implicit promotion" map(+, cunumeric_arr3, cunumeric_arr1)
        @test_throws DimensionMismatch cunumeric_arr1 .+ cunumeric_arr5
        @test_throws DimensionMismatch cunumeric_arr1 ./ cunumeric_arr5

        allowscalar() do
            @test cuNumeric.compare(
                cunumeric_arr1, cunumeric_int64 .+ cunumeric_arr1, atol(Float64), rtol(Float64)
            )
            r1 = @allowpromotion cunumeric_arr3 .+ cunumeric_arr1
            r2 = @allowpromotion map(+, cunumeric_arr3, cunumeric_arr1)
            @test cuNumeric.compare(r1, r2, atol(Float64), rtol(Float64))
        end
    end

    @testset "Copy-To" begin
        a = cuNumeric.zeros(2, 2)
        b = cuNumeric.ones(2, 2)
        copyto!(a, b);
        @test is_same(a, b)
    end
end

#TODO LOOP BINARY OPS WITH SCALARS
@testset verbose = true "Scalars" begin
    N = 10

    for T in Base.uniontypes(cuNumeric.SUPPORTED_NUMERIC_TYPES)
        julia_arr = my_rand(T, N)
        julia_arr_2D = my_rand(T, N, N)

        s = Random.rand(T)

        allowscalar() do
            cunumeric_arr = NDArray(julia_arr)
            cunumeric_arr_2D = NDArray(julia_arr_2D)
            allowpromotion(T == Int32) do
                for cn_arr in (cunumeric_arr, cunumeric_arr_2D)
                    @test cuNumeric.compare(s * julia_arr, s * cunumeric_arr, atol(T), rtol(T))
                    @test cuNumeric.compare(julia_arr * s, cunumeric_arr * s, atol(T), rtol(T))
                    @test cuNumeric.compare(s .* julia_arr, s .* cunumeric_arr, atol(T), rtol(T))
                    @test cuNumeric.compare(julia_arr .* s, cunumeric_arr .* s, atol(T), rtol(T))
                    @test cuNumeric.compare(s .+ julia_arr, s .+ cunumeric_arr, atol(T), rtol(T))
                    @test cuNumeric.compare(julia_arr .+ s, cunumeric_arr .+ s, atol(T), rtol(T))
                    @test cuNumeric.compare(s .- julia_arr, s .- cunumeric_arr, atol(T), rtol(T))
                    @test cuNumeric.compare(julia_arr .- s, cunumeric_arr .- s, atol(T), rtol(T))
                    @test cuNumeric.compare(s ./ julia_arr, s ./ cunumeric_arr, atol(T), rtol(T))
                    @test cuNumeric.compare(
                        s .* julia_arr .+ s, s .* cunumeric_arr .+ s, atol(T), rtol(T)
                    )
                    @test s + s ≈ (NDArray(s) + NDArray(s))[] # atol=atol(T) r_tol=atol(T)
                    # @test s * s ≈ (NDArray(s) * NDArray(s))[] # atol=atol(T) r_tol=atol(T)
                end
            end
        end
    end

    # Boolean things
    allowpromotion() do
        allowscalar() do
            julia_arr = [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            cunumeric_arr = NDArray(julia_arr)

            @test cuNumeric.compare(true * julia_arr, true * cunumeric_arr, atol(Int), rtol(Int))
            @test cuNumeric.compare(false * julia_arr, false * cunumeric_arr, atol(Int), rtol(Int))
            @test cuNumeric.compare(true .* julia_arr, true .* cunumeric_arr, atol(Int), rtol(Int))
            @test cuNumeric.compare(
                false .* julia_arr, false .* cunumeric_arr, atol(Int), rtol(Int)
            )

            julia_arr = [true, false, true, false, false, true, true]
            cunumeric_arr = NDArray(julia_arr)

            @test cuNumeric.compare(4 * julia_arr, 4 * cunumeric_arr, atol(Int), rtol(Int))
            @test cuNumeric.compare(4 .* julia_arr, 4 .* cunumeric_arr, atol(Int), rtol(Int))
            @test cuNumeric.compare(4 .+ julia_arr, 4 .+ cunumeric_arr, atol(Int), rtol(Int))
            @test cuNumeric.compare(
                julia_arr ./ 3, cunumeric_arr ./ 3, atol(Float64), rtol(Float64)
            )
        end
    end
end

@testset verbose = true "Powers" begin
    N = 9

    get_pwrs(::Type{I}) where {I<:Integer} = I.([-10, -5, -2, -1, 0, 1, 2, 5, 10])
    get_pwrs(::Type{F}) where {F<:AbstractFloat} = F.([-3.141, -2, -1, 0, 1, 2, 3.2, 4.41, 6.233])
    get_pwrs(::Type{Bool}) = [true, false, true, false, false, true, false, true, true]

    # TYPES = Base.uniontypes(cuNumeric.SUPPORTED_TYPES)
    TYPES = Base.uniontypes(cuNumeric.SUPPORTED_FLOAT_TYPES)

    @testset "$(BT) ^ $(PT)" for (BT, PT) in Iterators.product(TYPES, TYPES)
        base_jl = my_rand(BT, N)

        if BT <: Union{Bool,Int32} && PT == Int32
            # julia doesnt like Int32 powers
            pwrs = Float32.(get_pwrs(PT))
        elseif BT <: Union{Bool,Int32,Int64} && PT <: Union{Int32,Int64}
            # julia doesnt like Int64 powers on bool
            pwrs = Float64.(get_pwrs(PT))
        elseif (PT <: AbstractFloat) && (BT <: AbstractFloat || BT <: Signed)
            # Things like -387 ^ 3.2 will be Complex and error
            pwrs = get_pwrs(PT)
            base_jl = abs.(base_jl)
        else
            pwrs = get_pwrs(PT)
        end

        base_cn = @allowscalar NDArray(base_jl)
        pwrs_cn = @allowscalar NDArray(pwrs)

        # we deviate a bit form Julia here
        if (PT <: Union{Int32,Int64}) && (BT <: Union{Bool,Int32,Int64})
            T_OUT = cuNumeric.__my_promote_type(typeof(^), BT, PT)
        else
            T_OUT = Base.promote_op(Base.:(^), BT, PT)
        end

        TEST_BROKEN = (BT <: Union{Int32,Int64} && PT == Bool)

        allowpromotion(sizeof(BT) != sizeof(PT)) do
            allowscalar() do
                # Power is array
                @test cuNumeric.compare(
                    base_jl .^ pwrs, base_cn .^ pwrs_cn, atol(T_OUT), rtol(T_OUT)
                ) skip=TEST_BROKEN

                # Power is scalar
                for p in pwrs
                    @test cuNumeric.compare(base_jl .^ p, base_cn .^ p, atol(T_OUT), rtol(T_OUT))
                end
            end
        end
    end

    @testset verbose = true "Reciprocal" begin
        @testset for T in TYPES
            arr_jl = Random.rand(T, N)
            arr_cn = @allowscalar NDArray(arr_jl)

            # Differ from Julia here
            T_OUT = cuNumeric.__recip_type(T)

            # Cast julia result to whatever we do
            res_jl = T_OUT.(arr_jl .^ -1)
            allowpromotion(T == Bool || T == Int32) do
                res_cn = arr_cn .^ -1
                res_cn2 = inv.(arr_cn)
                allowscalar() do
                    @test cuNumeric.compare(res_jl, res_cn, atol(T_OUT), rtol(T_OUT))
                    @test cuNumeric.compare(res_jl, res_cn2, atol(T_OUT), rtol(T_OUT))
                end
            end
        end
    end

    @testset verbose = true "Square" begin
        @testset for T in TYPES
            arr_jl = Random.rand(T, N)
            arr_cn = @allowscalar NDArray(arr_jl)

            T_OUT = Base.promote_op(Base.:(^), T, Int64)
            res_jl = arr_jl .^ 2
            res_cn = arr_cn .^ 2

            allowscalar() do
                @test cuNumeric.compare(res_jl, res_cn, atol(T_OUT), rtol(T_OUT))
            end
        end
    end
end

@testset verbose = true "Slicing Tests" begin
    N = 100
    @testset for T in Base.uniontypes(cuNumeric.SUPPORTED_FLOAT_TYPES)
        slicing(T, N)
    end
end

@testset verbose = true "Scoping" begin
    N = 100

    @testset verbose = true for T in Base.uniontypes(cuNumeric.SUPPORTED_FLOAT_TYPES)
        allowscalar() do
            results = run_all_ops(T, N)
            for (name, (c_base, c_scoped)) in results
                @test cuNumeric.compare(c_base, c_scoped, atol(T), rtol(T))
            end

            u_rand = cuNumeric.as_type(cuNumeric.rand(NDArray, (15, 15)), T)
            v_rand = cuNumeric.as_type(cuNumeric.rand(NDArray, (15, 15)), T)

            u, v = gray_scott_base(T, N, u_rand, v_rand)
            u_scoped, v_scoped = gray_scott(T, N, u_rand, v_rand)

            @test cuNumeric.compare(u, u_scoped, atol(T) * N, rtol(T) * 10)
        end
    end
end

if run_gpu_tests
    include("tests/cuda/vecadd.jl")
    @testset verbose = true "CUDA Tests" begin
        cuda_unaryop(rtol(Float32))
        cuda_binaryop(rtol(Float32))
    end
else
    @warn "The CUDA tests will not be run as a CUDA-enabled device is not available"
end
