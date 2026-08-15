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

# Map functions to their required domains
const SPECIAL_DOMAINS = Dict(
    Base.acosh => :greater_than_one,
    Base.log => :positive,
    Base.log10 => :positive,
    Base.log2 => :positive,
    Base.log1p => :positive, # technicaly anything > -1
    Base.sqrt => :positive,
)

function test_unary_operation(func, julia_arr, cunumeric_arr, T)
    T_OUT = Base.promote_op(func, T)

    # Pre-allocate output arrays
    cunumeric_in_place = cuNumeric.zeros(T_OUT, size(julia_arr)...)

    # Compute results using different methods
    julia_res = func.(julia_arr)

    cunumeric_res = func.(cunumeric_arr)
    cunumeric_in_place .= func.(cunumeric_arr)
    cunumeric_res2 = map(func, cunumeric_arr)

    allowscalar() do
        @test safe_compare(julia_res, cunumeric_in_place, atol(T_OUT), rtol(T_OUT))
        @test safe_compare(julia_res, cunumeric_res, atol(T_OUT), rtol(T_OUT))
        @test safe_compare(julia_res, cunumeric_res2, atol(T_OUT), rtol(T_OUT))
    end
end

skip_on_integer = (
    Base.acosh, Base.atanh, Base.atan, Base.acos, Base.asin,
    Base.ceil, Base.floor, Base.trunc, Base.round, Base.signbit,
)
skip_on_bool = skip_on_integer
skip_on_complex = (
    Base.tanh,
    Base.deg2rad, Base.rad2deg, Base.sign, Base.cbrt,
    Base.exp2, Base.expm1, Base.log10, Base.log1p, Base.log2,
    Base.acos, Base.asin, Base.atan, Base.acosh, Base.asinh, Base.atanh,
    Base.ceil, Base.floor, Base.trunc, Base.signbit, Base.:(~),
)
skip_on_float = (Base.:(~),)

function test_unary_function_set(func_dict, T, N)
    default_generator = (T == Bool) ? :uniform : :unit_interval

    @testset "$func" for func in keys(func_dict)
        if func in skip_on_complex && (T <: Complex)
            continue
        end

        # The are only defined for like 3 integers (-1, 0, 1) so just skip them
        if func in skip_on_integer && (T <: Integer)
            continue
        end

        if func in skip_on_bool && (T == Bool)
            continue
        end

        if func in skip_on_float && (T <: AbstractFloat)
            continue
        end

        domain_type = get(SPECIAL_DOMAINS, func, default_generator)

        # :uniform is the only generator capable of generating bits
        skip = (T == Bool && domain_type != :uniform)
        skip && continue

        julia_arr_1D, julia_arr_2D = make_julia_arrays(T, N, domain_type)
        cunumeric_arr_1D, cunumeric_arr_2D = make_cunumeric_arrays(
            [julia_arr_1D], [julia_arr_2D], T, N
        )

        test_unary_operation(func, julia_arr_1D, cunumeric_arr_1D, T)
        test_unary_operation(func, julia_arr_2D, cunumeric_arr_2D, T)
    end
end

function test_unary_reduction_dims(
    func, julia_arr::AbstractArray{T,N}, cunumeric_arr::NDArray{T,N}
) where {T,N}
    allowpromotion(true) do
        for d in 1:N
            julia_res = func(julia_arr; dims=d)
            cunumeric_res = func(cunumeric_arr; dims=d)
            n = size(julia_arr, d)
            # Input magnitude for Higham-style absolute floor (see reduction_atol).
            scale = maximum(abs, julia_arr)
            allowscalar() do
                @test safe_compare(
                    julia_res, cunumeric_res, reduction_atol(T, n, scale), reduction_rtol(T, n)
                )
            end
        end

        # we are testing a multi axis reduction. This will throw a runtime error.
        # https://github.com/nv-legate/cupynumeric/blob/main/src/cupynumeric/ndarray.cc#L1132
        if N >= 2
            @test_throws Exception func(cunumeric_arr, dims=(1, 2))
        end
    end
end

function run_unary_tests(types; include_bool_reductions::Bool=false)
    @testset verbose = true "Unary Ops w/o Args" begin
        N = 100 # keep as perfect square

        @testset for T in types
            allowpromotion(true) do
                return test_unary_function_set(cuNumeric.floaty_unary_ops_no_args, T, N)
            end

            allowpromotion(T == Bool) do
                return test_unary_function_set(cuNumeric.unary_op_map_no_args, T, N)
            end

            if T <: AbstractFloat
                @testset "round is 1-arg only" begin
                    a = cuNumeric.ones(T, 4)
                    @test_throws ArgumentError round.(a; digits=1)
                end
            end
            # Special cases for unary ops that dont use . syntax
            @testset "- (Negation)" begin
                arr = my_rand(T, N)
                arr_cn = @allowscalar NDArray(arr)

                allowscalar() do
                    allowpromotion(T == Bool) do
                        T_OUT = T == Bool ? cuNumeric.DEFAULT_INT : T
                        @test safe_compare(T_OUT.(-arr), -arr_cn, atol(T), rtol(T))
                    end
                end
            end

            # Special cases for complex-related unary ops
            @testset "Complex Unary Ops (real, imag, conj)" begin
                if T <: Complex
                    arr = my_rand(T, N)
                    arr_cn = NDArray(arr)

                    allowscalar() do
                        allowpromotion(true) do
                            @test safe_compare(real(arr), real(arr_cn), atol(T), rtol(T))
                            @test safe_compare(imag(arr), imag(arr_cn), atol(T), rtol(T))
                            @test safe_compare(conj(arr), conj(arr_cn), atol(T), rtol(T))

                            @test safe_compare(real.(arr), real.(arr_cn), atol(T), rtol(T))
                            @test safe_compare(imag.(arr), imag.(arr_cn), atol(T), rtol(T))
                            @test safe_compare(conj.(arr), conj.(arr_cn), atol(T), rtol(T))
                        end
                    end
                end
            end
        end
    end

    @testset verbose = true "Unary Reductions" begin
        N = 100

        @testset for T in types
            julia_arr = my_rand(T, N)
            cunumeric_arr = @allowscalar NDArray(julia_arr)

            @testset "$(reduction)" for reduction in keys(cuNumeric.unary_reduction_map)
                # Skip reductions not supported by the cuNumeric backend for complex types
                if T <: Complex && (
                    reduction == Base.maximum ||
                    reduction == Base.minimum ||
                    reduction == Base.prod
                )
                    continue
                end

                allowpromotion(true) do
                    cunumeric_res = reduction(cunumeric_arr)
                    julia_res = reduction(julia_arr)

                    n = length(julia_arr)
                    scale = maximum(abs, julia_arr)
                    allowscalar() do
                        # assumes 0D result
                        @test isapprox(
                            julia_res, cunumeric_res[];
                            atol=reduction_atol(T, n, scale), rtol=reduction_rtol(T, n),
                        )
                    end
                end
            end
        end

        if include_bool_reductions
            # Test things that only work on Booleans
            julia_bools = rand(Bool, N)
            allowscalar() do
                cunumeric_bools = NDArray(julia_bools)
                @test any(julia_bools) == any(cunumeric_bools)[]
                @test all(julia_bools) == all(cunumeric_bools)[]
            end
        end
    end

    @testset verbose=true "Unary Reductions with Dims" begin
        N = 100

        @testset for T in types
            julia_arr_1D = my_rand(T, N)
            julia_arr_2D = my_rand(T, isqrt(N), isqrt(N))

            cunumeric_arr_1D = @allowscalar NDArray(julia_arr_1D)
            cunumeric_arr_2D = @allowscalar NDArray(julia_arr_2D)

            @testset "$(func)" for (func, _) in cuNumeric.unary_reduction_map
                # Skip reductions not supported by the cuNumeric backend for complex types
                if T <: Complex && (
                    func == Base.maximum ||
                    func == Base.minimum ||
                    func == Base.prod
                )
                    continue
                end

                ## TODO Int8 min/max along an axis is broken on GPU
                if cuNumeric.HAS_CUDA && T == Int8 && (func == Base.minimum || func == Base.maximum)
                    continue
                end

                test_unary_reduction_dims(func, julia_arr_1D, cunumeric_arr_1D)
                test_unary_reduction_dims(func, julia_arr_2D, cunumeric_arr_2D)
            end
        end
    end
end
