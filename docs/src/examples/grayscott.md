# Gray-Scott Reaction Diffusion

The runnable example in `examples/gray-scott.jl` evolves two chemical fields with periodic boundaries. Its update is a straight-line function, so the recommended function form of `@accelerate` can release reaction and Laplacian temporaries after their final use and fuse eligible CUDA broadcasts.

```julia
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
    #anim = Animation()

    N = 100
    dims = (N, N)

    args = Params()

    n_steps = 2000 # number of steps to take
    frame_interval = 200 # steps to take between making plots
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

        if n%frame_interval == 0
            u_cpu = u[:, :]
            heatmap(u_cpu, clims=(0, 1))
            frame(anim)
        end
    end
    gif(anim, "gray-scott.gif", fps=10)

    cuNumeric.h5write(SNAPSHOT_FILE, "u", snapshots)
    # h5write is asynchronous, so flush before another process opens the file.
    cuNumeric.Legate.runtime_sync()

    return u, v
 end
 ```

![Simulation Output](../gray-scott.gif)

The snapshots written to `gray-scott.h5` are the input to
[Dynamic Mode Decomposition](./dmd.md).
