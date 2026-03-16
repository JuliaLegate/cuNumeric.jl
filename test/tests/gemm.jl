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



function gemm(N, M, T, max_diff)

    if T == Bool
        a = cuNumeric.trues(5,5)
        b = cuNumeric.as_type(cuNumeric.trues(5,5), Float32)
        c = cuNumeric.as_type(cuNumeric.trues(5,5), Float64)
        @test_throws ArgumentError a * a # Bool * Bool not supported
        @allowpromotion d = a * b
        @allowpromotion e = a * c
        @test @allowscalar cuNumeric.compare(5 * ones(Float32, 5, 5), d, 0.0, max_diff)
        @test @allowscalar cuNumeric.compare(5 * ones(Float64, 5, 5), e, 0.0, max_diff)
        return
    end

    if T <: Integer
        a = cuNumeric.ones(Int32, 5, 5)
        a_jl = ones(Int32, 5, 5)
        b = a * a
        b_jl = a_jl * a_jl
        @test @allowscalar cuNumeric.compare(b_jl, b, 0.0, max_diff)
        return
    end

    dims_to_test = [(N,N), (N, M), (M, N)]

    @testset for dims in dims_to_test
        # Base julia arrays
        A_cpu = rand(T, dims[1], dims[2]);
        B_cpu = rand(T, dims[2], dims[1]);
        C_out_cpu = zeros(T, dims[1], dims[1])

        # cunumeric arrays
        A = cuNumeric.zeros(T, dims[1], dims[2])
        B = cuNumeric.zeros(T, dims[2], dims[1])
        C_out = cuNumeric.zeros(T, dims[1], dims[1]) 

        # Initialize NDArrays with random values
        # used in Julia arrays
        @allowscalar for i in 1:dims[1]
            for j in 1:dims[2]
                A[i, j] = A_cpu[i, j]
                B[j, i] = B_cpu[j, i]
            end
        end

        # Julia result
        C_cpu = A_cpu * B_cpu
        LinearAlgebra.mul!(C_out_cpu, A_cpu, B_cpu)

        @test C_cpu == C_out_cpu # really just making sure test is written right...

        A = cuNumeric.as_type(A, T)
        B = cuNumeric.as_type(B, T)
        C = cuNumeric.zeros(T, N, N)

        C = A * B
        LinearAlgebra.mul!(C_out, A, B)

        allowscalar() do
            @test isapprox(C, C_cpu, rtol = max_diff)
            @test isapprox(C, C_out, rtol = max_diff)

            if T != Float64
                C_wider = cuNumeric.zeros(Float64, dims[1], dims[1])
                @test_throws "Implicit promotion" LinearAlgebra.mul!(C_wider, A, B)
            end
        end

        # Integer output with FP input
        if !(T <: Integer)
            bad = cuNumeric.zeros(Int, dims[1], dims[1])
            @test_throws ArgumentError mul!(bad, A, B)
        end
    end
end
