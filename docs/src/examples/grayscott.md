# Gray-Scott Reaction Diffusion

The runnable example in `examples/gray-scott.jl` evolves two chemical fields with periodic boundaries. The update function uses `@accelerate` to fuse eligible CUDA broadcasts and
free temporary arrays after their last use.

```julia
using cuNumeric

struct Params{T}
    dx::T
    dt::T
    c_u::T
    c_v::T
    f::T
    k::T

    function Params(dx=1.0f0, c_u=1.0f0, c_v=0.3f0, f=0.03f0, k=0.06f0)
        return new{Float32}(dx, dx/5, c_u, c_v, f, k)
    end
end

function bc!(u_new, v_new, u, v)
    u_new[:, 1] = u[:, end - 1]
    u_new[:, end] = u[:, 2]
    u_new[1, :] = u[end - 1, :]
    u_new[end, :] = u[2, :]
    v_new[:, 1] = v[:, end - 1]
    v_new[:, end] = v[:, 2]
    v_new[1, :] = v[end - 1, :]
    return v_new[end, :] = v[2, :]
end

@accelerate function step!(u, v, u_new, v_new, args::Params)
    F_u = @. -u[2:end-1, 2:end-1] * v[2:end-1, 2:end-1]^2 +
        args.f * (1.0f0 - u[2:end-1, 2:end-1])
    F_v = @. u[2:end-1, 2:end-1] * v[2:end-1, 2:end-1]^2 -
        (args.f + args.k) * v[2:end-1, 2:end-1]

    u_lap = @. (
        (u[3:end, 2:end-1] - 2u[2:end-1, 2:end-1] + u[1:end-2, 2:end-1]) / args.dx^2 +
        (u[2:end-1, 3:end] - 2u[2:end-1, 2:end-1] + u[2:end-1, 1:end-2]) / args.dx^2
    )
    v_lap = @. (
        (v[3:end, 2:end-1] - 2v[2:end-1, 2:end-1] + v[1:end-2, 2:end-1]) / args.dx^2 +
        (v[2:end-1, 3:end] - 2v[2:end-1, 2:end-1] + v[2:end-1, 1:end-2]) / args.dx^2
    )

    u_new[2:end-1, 2:end-1] = @. (args.c_u * u_lap + F_u) * args.dt + u[2:end-1, 2:end-1]
    v_new[2:end-1, 2:end-1] = @. (args.c_v * v_lap + F_v) * args.dt + v[2:end-1, 2:end-1]
    bc!(u_new, v_new, u, v)
    return nothing
end

function gray_scott()

    N = 100
    dims = (N, N)

    args = Params()

    n_steps = 2000 # number of steps to take
    snapshot_interval = 20 # steps to take between saved snapshots

    u = cuNumeric.ones(dims)
    v = cuNumeric.zeros(dims)
    u_new = cuNumeric.zeros(dims)
    v_new = cuNumeric.zeros(dims)

    # One flattened frame per column, the layout DMD wants.
    snapshots = cuNumeric.zeros(Float32, N * N, n_steps ÷ snapshot_interval)

    u[1:15,1:15] = cuNumeric.rand(15,15)
    v[1:15,1:15] = cuNumeric.rand(15,15)

    for n in 1:n_steps
        step!(u, v, u_new, v_new, args)
        # update u and v
        # this doesn't copy, this switching references
        u, u_new = u_new, u
        v, v_new = v_new, v

        if n%snapshot_interval == 0
            snapshots[:, n ÷ snapshot_interval] = cuNumeric.reshape(u, (N * N, 1))
        end
    end

    cuNumeric.h5write("gray-scott.h5", "u", snapshots)

    return u, v
end

u, v = gray_scott()
```

![Simulation Output](../gray-scott.gif)
