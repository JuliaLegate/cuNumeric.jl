using cuNumeric

struct Params
    dx::Float32
    dt::Float32
    c_u::Float32
    c_v::Float32
    f::Float32
    k::Float32

    function Params(dx=1, c_u=1.0, c_v=0.3, f=0.03, k=0.06)
        new(dx, dx/5, c_u, c_v, f, k)
    end
end

function step(u, v, u_new, v_new, args::Params)
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

function gray_scott()
    N = 2000
    dims = (N, N)
    FT = Float32
    args = Params()

    n_steps = 1000 # number of steps to take

    u = cuNumeric.ones(FT, dims)
    v = cuNumeric.zeros(FT, dims)
    u_new = cuNumeric.zeros(FT, dims)
    v_new = cuNumeric.zeros(FT, dims)

    u[1:150, 1:150] = cuNumeric.as_type(cuNumeric.rand(NDArray, (150, 150)), FT)
    v[1:150, 1:150] = cuNumeric.as_type(cuNumeric.rand(NDArray, (150, 150)), FT)

    cuNumeric.disable_gc!()
    for n in 1:n_steps
        step(u, v, u_new, v_new, args)
        u, u_new = u_new, u
        v, v_new = v_new, v
    end
end

gray_scott()
