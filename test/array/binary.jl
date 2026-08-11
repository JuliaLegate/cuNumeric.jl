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

function test_binary_operation(func, julia_arr1, julia_arr2, cunumeric_arr1, cunumeric_arr2, T)
    T_OUT = Base.promote_op(func, T, T)

    # Pre-allocate output arrays
    cunumeric_in_place = cuNumeric.zeros(T_OUT, size(cunumeric_arr1)...)

    julia_res = func.(julia_arr1, julia_arr2)
    cunumeric_res = func.(cunumeric_arr1, cunumeric_arr2)
    cunumeric_in_place .= func.(cunumeric_arr1, cunumeric_arr2)
    cunumeric_res2 = map(func, cunumeric_arr1, cunumeric_arr2)

    allowscalar() do
        @test safe_compare(julia_res, cunumeric_in_place, atol(T_OUT), rtol(T_OUT))
        @test safe_compare(julia_res, cunumeric_res, atol(T_OUT), rtol(T_OUT))
        @test safe_compare(julia_res, cunumeric_res2, atol(T_OUT), rtol(T_OUT))
    end
end

function test_binary_function_set(func_dict, T, N)
    skip = (Base.lcm, Base.gcd)
    # not defined for complex.
    skip_on_complex = (
        Base.:(<), Base.:(<=), Base.:(>), Base.:(>=), Base.max, Base.min, Base.atan, Base.hypot
    )

    @testset "$func" for func in keys(func_dict)

        # This is tested separately
        func == Base.:(^) && continue

        if T <: Complex && (func in skip_on_complex)
            continue
        end

        (func in skip) && continue

        arrs_jl = make_julia_arrays(T, N, :uniform; count=2)
        arrs_cunum = make_cunumeric_arrays(arrs_jl[1:2], arrs_jl[3:4], T, N; count=2)

        test_binary_operation(func, arrs_jl[1:2]..., arrs_cunum[1:2]..., T)
        test_binary_operation(func, arrs_jl[3:4]..., arrs_cunum[3:4]..., T)
    end
end

@testset "Binary Ops" begin
    N = 100

    @testset for T in Base.uniontypes(cuNumeric.SUPPORTED_ARRAY_TYPES)
        allowpromotion(true) do
            test_binary_function_set(cuNumeric.floaty_binary_op_map, T, N)
            return test_binary_function_set(cuNumeric.binary_op_map, T, N)
        end

        arr_jl = my_rand(T, N)
        arr_jl2 = my_rand(T, N)
        arr_cn = @allowscalar NDArray(arr_jl)
        arr_cn2 = @allowscalar NDArray(arr_jl2)

        # lcm/gcd require specific handling for integers and avoid overflow
        if T <: cuNumeric.SUPPORTED_INT_TYPES && T != Bool
            range_limit = (T == Int8 || T == UInt8) ? 10 : 100
            arr_jl_small = my_rand(T, N; L=1, R=range_limit)
            arr_jl2_small = my_rand(T, N; L=1, R=range_limit)
            arr_cn_small = @allowscalar NDArray(arr_jl_small)
            arr_cn2_small = @allowscalar NDArray(arr_jl2_small)

            allowscalar() do
                @test cuNumeric.compare(
                    lcm.(arr_jl_small, arr_jl2_small), lcm.(arr_cn_small, arr_cn2_small), atol(T),
                    rtol(T),
                )
                @test cuNumeric.compare(
                    gcd.(arr_jl_small, arr_jl2_small), gcd.(arr_cn_small, arr_cn2_small), atol(T),
                    rtol(T),
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

@testset "Powers" begin
    N = 9

    get_pwrs(::Type{I}) where {I<:Integer} = I.([-10, -5, -2, -1, 0, 1, 2, 5, 10])
    get_pwrs(::Type{F}) where {F<:AbstractFloat} = F.([-3.141, -2, -1, 0, 1, 2, 3.2, 4.41, 6.233])
    get_pwrs(::Type{Bool}) = [true, false, true, false, false, true, false, true, true]

    # TYPES = Base.uniontypes(cuNumeric.SUPPORTED_ARRAY_TYPES)
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

        allowpromotion(true) do
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
            allowpromotion(true) do
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

            allowpromotion(true) do
                allowscalar() do
                    @test cuNumeric.compare(res_jl, res_cn, atol(T_OUT), rtol(T_OUT))
                end
            end
        end
    end
end

@testset "Copy-To" begin
    a = cuNumeric.zeros(2, 2)
    b = cuNumeric.ones(2, 2)
    copyto!(a, b)
    @test is_same(a, b)
end
