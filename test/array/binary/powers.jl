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
                @test safe_compare(
                    base_jl .^ pwrs, base_cn .^ pwrs_cn, atol(T_OUT), rtol(T_OUT)
                ) skip=TEST_BROKEN

                # Power is scalar
                for p in pwrs
                    @test safe_compare(base_jl .^ p, base_cn .^ p, atol(T_OUT), rtol(T_OUT))
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
                    @test safe_compare(res_jl, res_cn, atol(T_OUT), rtol(T_OUT))
                    @test safe_compare(res_jl, res_cn2, atol(T_OUT), rtol(T_OUT))
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
                    @test safe_compare(res_jl, res_cn, atol(T_OUT), rtol(T_OUT))
                end
            end
        end
    end
end
