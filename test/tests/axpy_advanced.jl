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

#= Purpose of test: daxpy_advanced
    -- add overloading support for [double/float scalar] * NDArray
    -- equavalence operator between a cuNumeric and Julia array without looping
    --          result == (α_cpu * x_cpu + y_cpu)
    --          (α_cpu * x_cpu + y_cpu) == 
    -- NDArray copy method allocates a new NDArray and copies all elements
    -- NDArray assign method assigns the contents from one NDArray to another NDArray
    -- x[:] colon notation for reading entire 1D NDArray to a Julia array 
    -- x[:, :] colon notation for reading entire 2D NDArray to a Julia array
    -- x[:, :] colon notation for filling entire 2D NDArray with scalar
    -- reshape method. we test a reshape from NxN to N*N
=#

function axpy_advanced(T, N)
    seed = 10
    dims = (N, N)

    α = T(56.6)

    # base Julia arrays
    x_cpu = Base.zeros(T, dims);
    y_cpu = Base.zeros(T, dims);

    # cunumeric arrays
    x = cuNumeric.zeros(T, dims)
    y = cuNumeric.zeros(T, dims)

    @test cuNumeric.dim(x) == 2
    @test cuNumeric.dim(y) == 2

    allowscalar() do
        @test is_same(x_cpu, x)
        @test is_same(y_cpu, y)
        @test is_same(x, x_cpu) # LHS and RHS are switched
        @test is_same(y, y_cpu)

        # test fill with scalar of all elements of the NDArray
        fill_value = T(4.23)
        fill!(x, fill_value)

        @test is_same(x, fill(fill_value, dims))

        ones_array = cuNumeric.ones(T, dims)
        ones_array_cpu = ones(T, dims)
        @test is_same(ones_array, ones_array_cpu)

        # create two random arrays
        x = cuNumeric.as_type(cuNumeric.random(Float64, size(x)), T)
        y = cuNumeric.as_type(cuNumeric.random(Float64, size(y)), T)

        # create a reference of NDArray
        x_ref = x
        y_ref = y
        @test is_same(x_ref, x)
        @test is_same(y_ref, y)

        # create a copy of NDArray
        x_copy = copy(x)
        y_copy = copy(y)
        @test is_same(x_copy, x)
        @test is_same(y_copy, y)

        # assign elements to a new array
        x_assign = cuNumeric.zeros(T, dims)
        y_assign = cuNumeric.zeros(T, dims)
        copyto!(x_assign, x)
        copyto!(y_assign, y)
        # lets check that it didn't assign with zeros
        # this is a check ensuring we didn't mess up the argument order
        @test !is_same(x_assign, cuNumeric.zeros(T, dims))
        @test !is_same(y_assign, cuNumeric.zeros(T, dims))
        # check the assigned values
        @test is_same(x_assign, x)
        @test is_same(y_assign, y)

        # set all the elements of each NDArray to the CPU 2D array equivalent
        x_cpu = Array(x)
        y_cpu = Array(y)
        @test is_same(x_cpu, x)
        @test is_same(y_cpu, y)

        # reshape a 2D array into 1D
        x_1d = cuNumeric.reshape(x, N * N)
        y_1d = cuNumeric.reshape(y, N * N)
        @test ndims(x_1d) == 1
        @test ndims(y_1d) == 1

        # set all the elements of each NDArray to the CPU 1D array equivalent
        x_cpu_1D = Array(x_1d)
        y_cpu_1D = Array(y_1d)
        @test is_same(x_cpu_1D, x_1d)
        @test is_same(y_cpu_1D, y_1d)

        result = α .* x .+ y

        # check results 
        @test is_same(result, (α * x_cpu + y_cpu))
        @test is_same(α * x_cpu + y_cpu, result) # LHS and RHS switched
    end
end
