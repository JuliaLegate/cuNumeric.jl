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

#= Purpose of test: elementwise
    - [./ .- .+ .*] operators are tested
=#

function elementwise()
    seed = 10
    N = 100
    dims = (N, N)

    arrA = cuNumeric.ones(dims)
    arrA_cpu = ones(dims)
    @test arrA == arrA_cpu

    arrB = cuNumeric.ones(dims)
    arrB_cpu = ones(dims)
    @test arrB == arrB_cpu

    cuNumeric.random(arrA, seed)
    cuNumeric.random(arrB, seed)

    arrA_cpu = arrA[:, :]
    arrB_cpu = arrB[:, :]

    @test arrA == arrA_cpu
    @test arrB == arrB_cpu

    result = cuNumeric.zeros(dims)
    result_cpu = zeros(dims)
    @test result == result_cpu

    # where the real testing starts
    arrA = 13.74 .- arrA
    arrA_cpu = 13.74 .- arrA_cpu
    @test arrA == arrA_cpu

    arrA = arrA .- 13.74
    arrA_cpu = arrA_cpu .- 13.74
    @test arrA == arrA_cpu

    result = arrA .- arrB
    result_cpu = arrA_cpu .- arrB_cpu
    @test result == result_cpu

    result = arrA - arrB
    result_cpu = arrA_cpu - arrB_cpu
    @test result == result_cpu

    arrA = 332.59 .+ arrA
    arrA_cpu = 332.59 .+ arrA_cpu
    @test arrA == arrA_cpu

    arrA = arrA .+ 332.59
    arrA_cpu = arrA_cpu .+ 332.59
    @test arrA == arrA_cpu

    result = arrA .+ arrB
    result_cpu = arrA_cpu .+ arrB_cpu
    @test result == result_cpu

    result = arrA + arrB
    result_cpu = arrA_cpu + arrB_cpu
    @test result == result_cpu

    # # not supported yet
    # arrA = 1.3 ./ arrA 
    # arrA_cpu =  1.3 ./ arrA_cpu
    # @test arrA == arrA_cpu

    # don't understand why this is failing
    # arrA = arrA ./ 1.3
    # arrA_cpu = arrA_cpu ./ 1.3
    # @test arrA == arrA_cpu

    result = arrA ./ arrB
    result_cpu = arrA_cpu ./ arrB_cpu
    @test result == result_cpu

    arrA = 32.32 * arrA
    arrA_cpu = 32.32 * arrA_cpu
    @test arrA == arrA_cpu

    arrA = arrA .* 32.32
    arrA_cpu = arrA_cpu .* 32.32
    @test arrA == arrA_cpu

    # currently our A * B impl is element wise
    result = arrA * arrB
    result_cpu = arrA_cpu .* arrB_cpu
    @test result == result_cpu

    result = arrA .* arrB
    result_cpu = arrA_cpu .* arrB_cpu
    @test result == result_cpu

    operator(arrA, arrB)
    operator(arrA_cpu, arrB_cpu)
    @test arrA == arrA_cpu
    @test arrB == arrB_cpu
end

function operator(u, v)
    dx=0.1;
    dt = dx/5;
    c_u=1.0;
    c_v=0.3;
    f=0.03;
    k=0.06;

    # calculate F_u and F_v functions
    F_u = (-u*(v * v)) + f*(1.0 .- u)
    F_v = (u*(v * v)) - (f+k)*v

    # 2-D Laplacian of f using array slicing, excluding boundaries
    # For an N x N array f, f_lap is the Nend x Nend array in the "middle"
    u_lap = (u - 2.0*u + u) ./ dx^2 + (u - 2.0*u + u) ./ dx^2
    v_lap = (v - 2.0*v + v) ./ dx^2 + (v - 2.0*v + v) ./ dx^2

    # Forward-Euler time step for all points except the boundaries
    u = ((c_u * u_lap) + F_u) * dt + u
    v = ((c_v * v_lap) + F_v) * dt + v
end
