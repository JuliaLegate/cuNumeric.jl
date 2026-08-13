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
```

Run it from the repository root with `julia --project=examples examples/gray-scott.jl`. The script converts only animation frames to host arrays.

![Simulation Output](../gray-scott.gif)
