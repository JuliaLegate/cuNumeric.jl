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

struct Params
    dx::Float64
    dt::Float64
    c_u::Float64
    c_v::Float64
    f::Float64
    k::Float64

    function Params(dx=0.1, c_u=1.0, c_v=0.3, f=0.03, k=0.06)
        new(dx, dx/5, c_u, c_v, f, k)
    end
end

function slicing(max_diff)
    N = 100
    dims = (N, N)

    FT = Float64
    args = Params()

    u = cuNumeric.zeros(Float64, dims)
    v = cuNumeric.zeros(Float64, dims)

    u_cpu = rand(FT, dims);
    v_cpu = rand(FT, dims);

    for i in 1:N
        for j in 1:N
            u[i, j] = Float64(u_cpu[i, j])
            v[i, j] = Float64(v_cpu[i, j])
        end
    end

    u_new = cuNumeric.zeros(Float64, dims)
    v_new = cuNumeric.zeros(Float64, dims)

    u_new_cpu = zeros(dims)
    v_new_cpu = zeros(dims)

    step(u_cpu, v_cpu, u_new_cpu, v_new_cpu, args)
    step(u, v, u_new, v_new, args)

    @test cuNumeric.compare(u, u_cpu, max_diff)
    @test cuNumeric.compare(v, v_cpu, max_diff)
    @test cuNumeric.compare(u_new, u_new_cpu, max_diff)
    @test cuNumeric.compare(v_new, v_new_cpu, max_diff)
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
    v_new[end, :] = v[2, :]
end
