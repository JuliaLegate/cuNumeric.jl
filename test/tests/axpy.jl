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
    -- Focused on double 2 dimenional. Does not test other types or dims. 
    -- NDArray intialization 
    -- NDArray writing and reading scalar indexing
    --          shows both [i, j] and [(i, j)] working
    -- NDArray addition and multiplication
=#
function axpy_basic()

    N = 100
    @testset for T in Base.uniontypes(cuNumeric.SUPPORTED_FLOAT_TYPES)
        α = T(56.6)
        dims = (N, N)

        # Base julia arrays
        x_cpu = rand(T, dims);
        y_cpu = rand(T, dims);

        # cunumeric arrays
        x = cuNumeric.zeros(T, dims)
        y = cuNumeric.zeros(T, dims)

        # Initialize NDArrays with same random values as julia arrays
        for i in 1:N
            for j in 1:N
                x[i, j] = x_cpu[i, j]
                y[i, j] = y_cpu[i, j]
            end
        end

        result = α .* x .+ y
        result_cpu = α .* x_cpu .+ y_cpu
        @test cuNumeric.compare(result, result_cpu, atol(T), rtol(T))
        @test cuNumeric.compare(result_cpu, result, atol(T), rtol(T))
    end
end
