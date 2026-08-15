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
    # Random data includes zeros / huge shift amounts; these are tested separately.
    skip = (
        Base.lcm, Base.gcd, Base.fld, Base.mod, Base.rem, Base.:(%), Base.:(<<), Base.:(>>)
    )
    # not defined for complex.
    skip_on_complex = (
        Base.:(<),
        Base.:(<=),
        Base.:(>),
        Base.:(>=),
        Base.max,
        Base.min,
        Base.atan,
        Base.hypot,
        Base.:(&),
        Base.:(|),
        Base.:(⊻),
        Base.copysign,
        Base.fld,
        Base.mod,
        Base.rem,
        Base.:(%),
        Base.:(<<),
        Base.:(>>),
    )
    skip_on_float = (Base.:(&), Base.:(|), Base.:(⊻), Base.:(<<), Base.:(>>))
    skip_on_integer = (Base.copysign,) # kernel is float-only; Bool <: Integer

    @testset "$func" for func in keys(func_dict)

        # This is tested separately
        func == Base.:(^) && continue

        if T <: Complex && (func in skip_on_complex)
            continue
        end

        if T <: AbstractFloat && (func in skip_on_float)
            continue
        end

        if T <: Integer && (func in skip_on_integer)
            continue
        end

        (func in skip) && continue

        arrs_jl = make_julia_arrays(T, N, :uniform; count=2)
        arrs_cunum = make_cunumeric_arrays(arrs_jl[1:2], arrs_jl[3:4], T, N; count=2)

        test_binary_operation(func, arrs_jl[1:2]..., arrs_cunum[1:2]..., T)
        test_binary_operation(func, arrs_jl[3:4]..., arrs_cunum[3:4]..., T)
    end
end

function run_binary_ops_tests(types)
    @testset "Binary Ops" begin
        N = 100

        @testset for T in types
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
                    @test safe_compare(
                        lcm.(arr_jl_small, arr_jl2_small), lcm.(arr_cn_small, arr_cn2_small),
                        atol(T),
                        rtol(T),
                    )
                    @test safe_compare(
                        gcd.(arr_jl_small, arr_jl2_small), gcd.(arr_cn_small, arr_cn2_small),
                        atol(T),
                        rtol(T),
                    )
                end
            end

            # fld/mod/rem/% need a non-zero divisor. FLOOR_DIVIDE is fld, not div.
            if (T <: cuNumeric.SUPPORTED_INT_TYPES && T != Bool) || T <: AbstractFloat
                if T <: Integer
                    arr_jl_div = my_rand(T, N; L=(T <: Unsigned ? 1 : -20), R=20)
                    arr_jl_den = my_rand(T, N; L=1, R=20)
                else
                    arr_jl_div = my_rand(T, N)
                    arr_jl_den = my_rand(T, N)
                    arr_jl_den = map(
                        x -> abs(x) < T(1) ? copysign(one(T), iszero(x) ? one(T) : x) : x,
                        arr_jl_den,
                    )
                end
                arr_cn_div = @allowscalar NDArray(arr_jl_div)
                arr_cn_den = @allowscalar NDArray(arr_jl_den)

                allowscalar() do
                    for func in (fld, mod, rem, Base.:%)
                        @test safe_compare(
                            func.(arr_jl_div, arr_jl_den), func.(arr_cn_div, arr_cn_den),
                            atol(T), rtol(T),
                        )
                    end
                end
            end

            # Shifts: small non-negative amounts. Left shift uses a small lhs to
            # avoid C++ undefined behavior on signed overflow.
            if T <: cuNumeric.SUPPORTED_INT_TYPES && T != Bool
                max_shift = T(min(7, 8 * sizeof(T) - 1))
                arr_jl_sh = my_rand(T, N; L=0, R=max_shift)
                arr_jl_lshift_lhs = my_rand(T, N; L=0, R=7)
                arr_jl_rshift_lhs = my_rand(T, N)
                arr_cn_sh = @allowscalar NDArray(arr_jl_sh)
                arr_cn_lshift_lhs = @allowscalar NDArray(arr_jl_lshift_lhs)
                arr_cn_rshift_lhs = @allowscalar NDArray(arr_jl_rshift_lhs)

                allowscalar() do
                    @test safe_compare(
                        arr_jl_lshift_lhs .<< arr_jl_sh, arr_cn_lshift_lhs .<< arr_cn_sh,
                        atol(T), rtol(T),
                    )
                    @test safe_compare(
                        arr_jl_rshift_lhs .>> arr_jl_sh, arr_cn_rshift_lhs .>> arr_cn_sh,
                        atol(T), rtol(T),
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
end

function run_binary_copyto_tests()
    @testset "Copy-To" begin
        a = cuNumeric.zeros(2, 2)
        b = cuNumeric.ones(2, 2)
        copyto!(a, b)
        @test is_same(a, b)
    end
end
