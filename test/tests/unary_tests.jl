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
        @test cuNumeric.compare(julia_res, cunumeric_in_place, atol(T_OUT), rtol(T_OUT))
        @test cuNumeric.compare(julia_res, cunumeric_res, atol(T_OUT), rtol(T_OUT))
        @test cuNumeric.compare(julia_res, cunumeric_res2, atol(T_OUT), rtol(T_OUT))
    end
end

skip_on_integer = (Base.acosh, Base.atanh, Base.atan, Base.acos, Base.asin)
skip_on_bool = (Base.:(-), skip_on_integer...)
skip_on_complex = (
    Base.tanh,
    Base.deg2rad, Base.rad2deg, Base.sign, Base.cbrt,
    Base.exp2, Base.expm1, Base.log10, Base.log1p, Base.log2,
    Base.acos, Base.asin, Base.atan, Base.acosh, Base.asinh, Base.atanh,
)

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
