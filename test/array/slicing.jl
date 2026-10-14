#= Copyright 2025 Northwestern University,
 *                   Carnegie Mellon University University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSEend-2.0
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

#= Purpose of test: slicing
    -- Perform NDArray operations using slices for more efficient memory access
=#

struct Params{T}
    dx::T
    dt::T
    c_u::T
    c_v::T
    f::T
    k::T

    function Params(
        ::Type{T}, dx=T(0.1), c_u=T(1.0), c_v=T(0.3), f=T(0.03), k=T(0.06)
    ) where {T<:AbstractFloat}
        return new{T}(dx, dx/T(5), c_u, c_v, f, k)
    end
end

function slicing(T, N)
    dims = (N, N)

    args = Params(T)

    u = cuNumeric.zeros(T, dims)
    v = cuNumeric.zeros(T, dims)

    u_cpu = rand(T, dims)
    v_cpu = rand(T, dims)

    @allowscalar for i in 1:N
        for j in 1:N
            u[i, j] = T(u_cpu[i, j])
            v[i, j] = T(v_cpu[i, j])
        end
    end

    u_new = cuNumeric.zeros(T, dims)
    v_new = cuNumeric.zeros(T, dims)

    u_new_cpu = zeros(T, dims)
    v_new_cpu = zeros(T, dims)

    step(u_cpu, v_cpu, u_new_cpu, v_new_cpu, args)
    step(u, v, u_new, v_new, args)

    allowscalar() do
        @test safe_compare(u, u_cpu, atol(T), rtol(T))
        @test safe_compare(v, v_cpu, atol(T), rtol(T))
        @test safe_compare(u_new, u_new_cpu, atol(T), rtol(T))
        @test safe_compare(v_new, v_new_cpu, atol(T), rtol(T))
    end
end

# gray scott
function step(u, v, u_new, v_new, args::Params)
    # calculate F_u and F_v functions
    # currently we don't have NDArray^x working yet.
    F_u = (
        (
            -u[2:(end - 1), 2:(end - 1)] .*
            (v[2:(end - 1), 2:(end - 1)] .* v[2:(end - 1), 2:(end - 1)])
        ) +
        args.f*(1 .- u[2:(end - 1), 2:(end - 1)])
    )
    F_v = (
        (
            u[2:(end - 1), 2:(end - 1)] .*
            (v[2:(end - 1), 2:(end - 1)] .* v[2:(end - 1), 2:(end - 1)])
        ) -
        (args.f+args.k)*v[2:(end - 1), 2:(end - 1)]
    )
    # 2-D Laplacian of f using array slicing, excluding boundaries
    # For an N x N array f, f_lap is the Nend x Nend array in the "middle"
    u_lap = (
        (u[3:end, 2:(end - 1)] - 2*u[2:(end - 1), 2:(end - 1)]
         +
         u[1:(end - 2), 2:(end - 1)]) ./ args.dx^2
        +
        (u[2:(end - 1), 3:end] - 2*u[2:(end - 1), 2:(end - 1)]
         +
         u[2:(end - 1), 1:(end - 2)]) ./ args.dx^2
    )
    v_lap = (
        (v[3:end, 2:(end - 1)] - 2*v[2:(end - 1), 2:(end - 1)] + v[1:(end - 2), 2:(end - 1)]) ./
        args.dx^2
        +
        (v[2:(end - 1), 3:end] - 2*v[2:(end - 1), 2:(end - 1)] + v[2:(end - 1), 1:(end - 2)]) ./
        args.dx^2
    )

    # Forward-Euler time step for all points except the boundaries
    u_new[2:(end - 1), 2:(end - 1)] =
        ((args.c_u * u_lap) + F_u) * args.dt + u[2:(end - 1), 2:(end - 1)]
    v_new[2:(end - 1), 2:(end - 1)] =
        ((args.c_v * v_lap) + F_v) * args.dt + v[2:(end - 1), 2:(end - 1)]

    # Apply periodic boundary conditions
    u_new[:, 1] = u[:, end - 1]
    u_new[:, end] = u[:, 2]
    u_new[1, :] = u[end - 1, :]
    u_new[end, :] = u[2, :]
    v_new[:, 1] = v[:, end - 1]
    v_new[:, end] = v[:, 2]
    v_new[1, :] = v[end - 1, :]
    return v_new[end, :] = v[2, :]
end

@testset "Array Slices" begin
    N = 100
    @testset for T in Base.uniontypes(cuNumeric.SUPPORTED_FLOAT_TYPES)
        slicing(T, N)
    end
end

@testset "Basic Array Accessors" begin
    for T in Base.uniontypes(cuNumeric.SUPPORTED_ARRAY_TYPES)
        value = rand(T)
        arr = cuNumeric.zeros(T, 2, 2)

        allowscalar() do
            arr[1, 2] = value
            @test arr[1, 2] == value
        end
    end
end

# Issue #211
@testset "Range assignment" begin
    A = NDArray(Float32[1, 2, 3, 4])
    A[2:3] = NDArray(Float32[10, 20])
    @test Array(A) == Float32[1, 10, 20, 4]
    A[1:4] = NDArray(Float32[5, 6, 7, 8])
    @test Array(A) == Float32[5, 6, 7, 8]
    A[3:2] = NDArray(Float32[])
    @test Array(A) == Float32[5, 6, 7, 8]
    A[2:3] = NDArray(Int64[1, 2])
    @test Array(A) == Float32[5, 1, 2, 8]
    @test_throws DimensionMismatch (A[2:3] = NDArray(Float32[1, 2, 3]))
    @test_throws BoundsError (A[0:1] = NDArray(Float32[1, 2]))

    M = NDArray(Float32[1 2; 3 4; 5 6])
    M[2:3, :] = NDArray(Float32[30 40; 50 60])
    @test Array(M) == Float32[1 2; 30 40; 50 60]
    M[1, :] = NDArray(Float32[7, 8])
    M[:, 2] = NDArray(Float32[0, 0, 0])
    M[2:3, 1] = NDArray(Float32[9, 9])
    @test Array(M) == Float32[7 0; 9 0; 9 0]
    @test_throws DimensionMismatch (M[2:3, :] = NDArray(Float32[1 2 3; 4 5 6]))
    @test_throws DimensionMismatch (M[:, 1] = NDArray(Float32[1, 2]))
    @test Array(M) == Float32[7 0; 9 0; 9 0]
end
