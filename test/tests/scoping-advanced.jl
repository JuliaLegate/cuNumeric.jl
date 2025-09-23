using cuNumeric

struct ParamsGS{T<:AbstractFloat}
    dx::T
    dt::T
    c_u::T
    c_v::T
    f::T
    k::T

    # Constructor with default values
    function ParamsGS{T}(
        dx::T=one(T), c_u::T=one(T), c_v::T=T(0.3), f::T=T(0.03), k::T=T(0.06)
    ) where {T<:AbstractFloat}
        new(dx, dx/5, c_u, c_v, f, k)
    end
end

function step(u, v, u_new, v_new, args::ParamsGS)
    @cunumeric begin
        # calculate F_u and F_v functions
        # currently we don't have NDArray^x working yet. 
        F_u = (
            (
                -u[2:(end - 1), 2:(end - 1)] .*
                (v[2:(end - 1), 2:(end - 1)] .* v[2:(end - 1), 2:(end - 1)])
            ) + args.f*(1 .- u[2:(end - 1), 2:(end - 1)])
        )
        F_v = (
            (
                u[2:(end - 1), 2:(end - 1)] .*
                (v[2:(end - 1), 2:(end - 1)] .* v[2:(end - 1), 2:(end - 1)])
            ) - (args.f+args.k)*v[2:(end - 1), 2:(end - 1)]
        )
        # 2-D Laplacian of f using array slicing, excluding boundaries
        # For an N x N array f, f_lap is the Nend x Nend array in the "middle"
        u_lap = (
            (
                u[3:end, 2:(end - 1)] - 2*u[2:(end - 1), 2:(end - 1)] +
                u[1:(end - 2), 2:(end - 1)]
            ) ./ args.dx^2 +
            (
                u[2:(end - 1), 3:end] - 2*u[2:(end - 1), 2:(end - 1)] +
                u[2:(end - 1), 1:(end - 2)]
            ) ./ args.dx^2
        )
        v_lap = (
            (
                v[3:end, 2:(end - 1)] - 2*v[2:(end - 1), 2:(end - 1)] +
                v[1:(end - 2), 2:(end - 1)]
            ) ./ args.dx^2 +
            (
                v[2:(end - 1), 3:end] - 2*v[2:(end - 1), 2:(end - 1)] +
                v[2:(end - 1), 1:(end - 2)]
            ) ./ args.dx^2
        )

        # # Forward-Euler time step for all points except the boundaries
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
end

# same as above but without @cunumeric macro
function step_base(u, v, u_new, v_new, args::ParamsGS)
    # calculate F_u and F_v functions
    # currently we don't have NDArray^x working yet. 
    F_u = (
        (
            -u[2:(end - 1), 2:(end - 1)] .*
            (v[2:(end - 1), 2:(end - 1)] .* v[2:(end - 1), 2:(end - 1)])
        ) + args.f*(1 .- u[2:(end - 1), 2:(end - 1)])
    )
    F_v = (
        (
            u[2:(end - 1), 2:(end - 1)] .*
            (v[2:(end - 1), 2:(end - 1)] .* v[2:(end - 1), 2:(end - 1)])
        ) - (args.f+args.k)*v[2:(end - 1), 2:(end - 1)]
    )
    # 2-D Laplacian of f using array slicing, excluding boundaries
    # For an N x N array f, f_lap is the Nend x Nend array in the "middle"
    u_lap = (
        (
            u[3:end, 2:(end - 1)] - 2*u[2:(end - 1), 2:(end - 1)] +
            u[1:(end - 2), 2:(end - 1)]
        ) ./ args.dx^2 +
        (
            u[2:(end - 1), 3:end] - 2*u[2:(end - 1), 2:(end - 1)] +
            u[2:(end - 1), 1:(end - 2)]
        ) ./ args.dx^2
    )
    v_lap = (
        (
            v[3:end, 2:(end - 1)] - 2*v[2:(end - 1), 2:(end - 1)] +
            v[1:(end - 2), 2:(end - 1)]
        ) ./ args.dx^2 +
        (
            v[2:(end - 1), 3:end] - 2*v[2:(end - 1), 2:(end - 1)] +
            v[2:(end - 1), 1:(end - 2)]
        ) ./ args.dx^2
    )

    # # Forward-Euler time step for all points except the boundaries
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

function gray_scott(FT, n_steps, u_rand, v_rand)
    N = 100
    dims = (N, N)
    args = ParamsGS{FT}()
    u = cuNumeric.ones(FT, dims)
    v = cuNumeric.zeros(FT, dims)
    u_new = cuNumeric.zeros(FT, dims)
    v_new = cuNumeric.zeros(FT, dims)

    u[1:15, 1:15] = u_rand
    v[1:15, 1:15] = v_rand

    cuNumeric.disable_gc!()
    for n in 1:n_steps
        step(u, v, u_new, v_new, args)
        u, u_new = u_new, u
        v, v_new = v_new, v
    end

    return u, v
end

function gray_scott_base(FT, n_steps, u_rand, v_rand)
    N = 100
    dims = (N, N)
    args = ParamsGS{FT}()
    u = cuNumeric.ones(FT, dims)
    v = cuNumeric.zeros(FT, dims)
    u_new = cuNumeric.zeros(FT, dims)
    v_new = cuNumeric.zeros(FT, dims)

    u[1:15, 1:15] = u_rand
    v[1:15, 1:15] = v_rand

    cuNumeric.init_gc!()
    for n in 1:n_steps
        step_base(u, v, u_new, v_new, args)
        u, u_new = u_new, u
        v, v_new = v_new, v
    end

    return u, v
end
