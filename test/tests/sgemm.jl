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

#= Purpose of test: daxpy
    - Test a matrix matrix multiply
=#
function sgemm(max_diff)
    N = 100
    FT = Float32

    dims = (N, N)

    # Base julia arrays
    A_cpu = rand(FT, dims);
    B_cpu = rand(FT, dims);

    # cunumeric arrays
    A = cuNumeric.zeros(dims)
    B = cuNumeric.zeros(dims)

    # Initialize NDArrays with random values
    # used in Julia arrays
    for i in 1:N
        for j in 1:N
            A[i, j] = Float64(A_cpu[i, j])
            B[i, j] = Float64(B_cpu[i, j])
        end
    end

    # Needed to start as Float64 to 
    # initialize the NDArray
    C_cpu = A_cpu .* B_cpu

    A = cuNumeric.as_type(A, FT)
    B = cuNumeric.as_type(B, FT)
    C = cuNumeric.zeros(FT, N, N)

    C = A * B

    @test cuNumeric.compare(C, C_cpu, max_diff)
    # same comparison as above using approx
    @test C ≈ C_cpu atol=max_diff
end
