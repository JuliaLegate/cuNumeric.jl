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
