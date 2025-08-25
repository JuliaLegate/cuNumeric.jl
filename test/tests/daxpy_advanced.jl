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

function daxpy_advanced()
    seed = 10

    N = 100
    dims = (N, N)

    α = 56.6

    # base Julia arrays
    x_cpu = Base.zeros(Float64, dims);
    y_cpu = Base.zeros(Float64, dims);

    # cunumeric arrays
    x = cuNumeric.zeros(Float64, dims)
    y = cuNumeric.zeros(Float64, dims)

    @test cuNumeric.dim(x) == 2
    @test cuNumeric.dim(y) == 2

    @test x_cpu == x
    @test y_cpu == y
    @test x == x_cpu # LHS and RHS are switched
    @test y == y_cpu

    # test fill with scalar of all elements of the NDArray
    fill_value = Float64(4.23)
    x[:, :] = fill_value

    @test x == fill(fill_value, dims)

    ones_array = cuNumeric.ones(Float64, dims)
    ones_array_cpu = ones(Float64, dims)
    @test ones_array == ones_array_cpu

    # create two random arrays
    cuNumeric.random(x, seed)
    cuNumeric.random(y, seed)

    # create a reference of NDArray
    x_ref = x
    y_ref = y
    @test x_ref == x
    @test y_ref == y

    # create a copy of NDArray
    x_copy = copy(x)
    y_copy = copy(y)
    @test x_copy == x
    @test y_copy == y

    # assign elements to a new array
    x_assign = cuNumeric.zeros(Float64, dims)
    y_assign = cuNumeric.zeros(Float64, dims)
    cuNumeric.assign(x_assign, x)
    cuNumeric.assign(y_assign, y)
    # lets check that it didn't assign with zeros
    # this is a check ensuring we didn't mess up the argument order
    @test x_assign != cuNumeric.zeros(Float64, dims)
    @test y_assign != cuNumeric.zeros(Float64, dims)
    # check the assigned values
    @test x_assign == x
    @test y_assign == y

    # set all the elements of each NDArray to the CPU 2D array equivalent
    x_cpu = x[:, :]
    y_cpu = y[:, :]
    @test x_cpu == x
    @test y_cpu == y

    # reshape a 2D array into 1D
    x_1d = cuNumeric.reshape(x, N * N)
    y_1d = cuNumeric.reshape(y, N * N)
    @test ndims(x_1d) == 1
    @test ndims(y_1d) == 1

    # set all the elements of each NDArray to the CPU 1D array equivalent
    x_cpu_1D = x_1d[:]
    y_cpu_1D = y_1d[:]
    @test x_cpu_1D == x_1d
    @test y_cpu_1D == y_1d

    result = α .* x .+ y

    # check results 
    @test result == (α * x_cpu + y_cpu)
    @test (α * x_cpu + y_cpu) == result # LHS and RHS switched

    nothing
end
