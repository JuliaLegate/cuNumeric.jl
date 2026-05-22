using cuNumeric
using Random
using StatsBase

cuNumeric.versioninfo()

Random.seed!(1234)

struct Params{T}
    dx::T
    dt::T
    c_u::T
    c_v::T
    f::T
    k::T

    function Params(dx=1.0f0, c_u=1.0f0, c_v=0.3f0, f=0.03f0, k=0.06f0)
        new{Float32}(dx, dx/5, c_u, c_v, f, k)
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
    v_new[end, :] = v[2, :]
end

function step!(u, v, u_new, v_new, args::Params)

    dx_sq = args.dx^2

    # AVOID SOME ISSUES WITH ^ operator
    v_sq = v[2:(end - 1), 2:(end - 1)] .* v[2:(end - 1), 2:(end - 1)]

    # calculate F_u and F_v functions
    F_u = (
        (-u[2:(end - 1), 2:(end - 1)] .* (v_sq)) .+
        args.f*(1.0f0 .- u[2:(end - 1), 2:(end - 1)])
    )
    F_v = (
        (u[2:(end - 1), 2:(end - 1)] .* (v_sq)) .-
        (args.f+args.k) .* v[2:(end - 1), 2:(end - 1)]
    )
    # 2-D Laplacian of f using array slicing, excluding boundaries
    # For an N x N array f, f_lap is the Nend x Nend array in the "middle"
    u_lap = (
        (u[3:end, 2:(end - 1)] - 2*u[2:(end - 1), 2:(end - 1)] + u[1:(end - 2), 2:(end - 1)]) ./
        dx_sq
        +
        (u[2:(end - 1), 3:end] - 2*u[2:(end - 1), 2:(end - 1)] + u[2:(end - 1), 1:(end - 2)]) ./
        dx_sq
    )
    v_lap = (
        (v[3:end, 2:(end - 1)] - 2*v[2:(end - 1), 2:(end - 1)] + v[1:(end - 2), 2:(end - 1)]) ./
        dx_sq
        +
        (v[2:(end - 1), 3:end] - 2*v[2:(end - 1), 2:(end - 1)] + v[2:(end - 1), 1:(end - 2)]) ./
        dx_sq
    )

    # Forward-Euler time step for all points except the boundaries
    u_new[2:(end - 1), 2:(end - 1)] =
        ((args.c_u * u_lap) + F_u) * args.dt + u[2:(end - 1), 2:(end - 1)]
    v_new[2:(end - 1), 2:(end - 1)] =
        ((args.c_v * v_lap) + F_v) * args.dt + v[2:(end - 1), 2:(end - 1)]

    # Apply periodic boundary conditions
    bc!(u_new, v_new, u, v)
end

function initialize(N)
    u = cuNumeric.ones(N, N)
    v = cuNumeric.zeros(N, N)
    u_new = cuNumeric.zeros(N, N)
    v_new = cuNumeric.zeros(N, N)

    rand_frac = round(Int, 0.15 * N)

    # Make julia arrays first for reproducability, I don't think
    # we have a way to set cuNumeric's random seed.
    u_tmp = rand(Float32, rand_frac, rand_frac)
    v_tmp = rand(Float32, rand_frac, rand_frac)

    # Copy over values to cuNumeric arrays
    allowscalar() do
        for i in 1:rand_frac, j in 1:rand_frac
            u[i,j] = u_tmp[i,j]
            v[i,j] = v_tmp[i,j]
        end
    end

    GC.gc() # remove any intermediates

    return u, v, u_new, v_new
end

# Re-iplement loop without plotting, and add timing
function gray_scott(N::Int, n_steps::Int, n_warmup::Int)

    args = Params()
    u, v, u_new, v_new = initialize(N)
    @info "Initialized Arrays"

    for n in 1:n_warmup
        @info "Warmup Step $n / $n_warmup"
        step!(u, v, u_new, v_new, args)
        u, u_new = u_new, u
        v, v_new = v_new, v
    end

    step_times_us = Vector{Float64}(undef, n_steps)
    for n in 1:n_steps
        @info "Step $n / $n_steps"
        start_time = get_time_microseconds()
        step!(u, v, u_new, v_new, args)
        u, u_new = u_new, u
        v, v_new = v_new, v
        step_times_us[n] = get_time_microseconds() - start_time

        if n%10 == 0
            GC.gc()
        end
    end

    return u, v, step_times_us
end

gpus = parse(Int, ARGS[1])
N = parse(Int, ARGS[2])
n_steps = parse(Int, ARGS[3])
n_warmup = parse(Int, ARGS[4])

println(
    "[cuNumeric]  Gray-Scott benchmark on $(N)x$(N) grid for $(n_steps) iterations, $(n_warmup) warmups"
)

u, v, step_times_us = gray_scott(N, n_steps, n_warmup)

step_times_ms = step_times_us ./ 1000.0
mean_time_ms = mean(step_times_ms) 
std_time_ms = std(step_times_ms)

total_time_s = sum(step_times_ms) / 1000.0
iter_per_second = n_steps / total_time_s

println("Mean Step Time: $(mean_time_ms) ms")
println("StdDev Step Time: $(std_time_ms) ms")
println("Min Step Time: $(minimum(step_times_ms)) ms")
println("Max Step Time: $(maximum(step_times_ms)) ms")
println("Throughput: $(iter_per_second) ")

# open("./gray_scott.csv", "a") do io
#     @printf(io, "%s,%d,%d,%d,%.6f,%.6f\n", "cunumeric", gpus, N, M, mean_time_ms, gflops)
# end