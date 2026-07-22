struct GSParams{T}
    dx::T
    dt::T
    c_u::T
    c_v::T
    f::T
    k::T
end

function GSParams{T}(; dx=1, c_u=1.0, c_v=0.3, f=0.03, k=0.06) where {T}
    return GSParams{T}(T(dx), T(dx / 5), T(c_u), T(c_v), T(f), T(k))
end

abstract type AbstractGrayScott{T} <: AbstractBenchmark{T} end

Base.@kwdef struct GrayScottBaseline{T} <: AbstractGrayScott{T}
    N::Int
    M::Int
end

Base.@kwdef struct GrayScottLifetimes{T} <: AbstractGrayScott{T}
    N::Int
    M::Int
end

name(::AbstractGrayScott) = "grayscott"
dims(b::AbstractGrayScott) = (b.N, b.M)
data(b::AbstractGrayScott{T}) where {T} = "GrayScott with T=$(T), N=$(b.N), M=$(b.M)"
allowed_types(::Type{AbstractGrayScott}) = cuNumeric.SUPPORTED_FLOAT_TYPES
total_flops(b::AbstractGrayScott) = b.N * b.M # grid points updated per step

function build_benchmark(::Type{A}, ::Type{T}, N, M) where {A<:AbstractGrayScott,T}
    return A{T}(; N=N, M=M)
end

mutable struct GrayScottState{A,P}
    u::A
    v::A
    u_new::A
    v_new::A
    params::P
end

function initialize(b::AbstractGrayScott{T}; mod=cuNumeric, deterministic::Bool=false) where {T}
    d = (b.N, b.M)
    u = mod.ones(T, d)
    v = mod.zeros(T, d)
    u_new = mod.zeros(T, d)
    v_new = mod.zeros(T, d)

    seed = min(150, b.N, b.M)
    if deterministic
        # Fixed host pattern so CPU and GPU (any GPU count) share the same IC.
        # Avoids Random streams differing across array backends.
        host_u = T[
            T(0.5) + T(0.5) * sin(T(i)) * cos(T(j)) for i in 1:seed, j in 1:seed
        ]
        host_v = T[
            T(0.25) + T(0.25) * cos(T(i)) * sin(T(j)) for i in 1:seed, j in 1:seed
        ]
        u[1:seed, 1:seed] = mod === cuNumeric ? NDArray(host_u) : host_u
        v[1:seed, 1:seed] = mod === cuNumeric ? NDArray(host_v) : host_v
    else
        u[1:seed, 1:seed] = mod.rand(T, (seed, seed))
        v[1:seed, 1:seed] = mod.rand(T, (seed, seed))
    end

    return (GrayScottState(u, v, u_new, v_new, GSParams{T}()),)
end

correctness_supported(::AbstractGrayScott) = true

function check_benchmark_correctness(
    b::AbstractGrayScott{T}, gs::GlobalSettings; mod=cuNumeric, atol=1e-4, rtol=1e-4
) where {T}
    # CPU reference compares via cuNumeric.compare (scalar gather). Other backends skip.
    mod === cuNumeric || return "skipped"

    n = gs.n_correctness_iter
    st_gpu = only(initialize(b; mod=mod, deterministic=true))
    st_cpu = only(initialize(b; mod=Base, deterministic=true))

    for _ in 1:n
        run!(b, st_gpu)
        run!(b, st_cpu)
    end

    # Element-wise NDArray indexing gathers across tiles — do not use Array(NDArray)
    # for multi-GPU (get_ptr is local-tile only).
    u_ok = @allowscalar cuNumeric.compare(st_cpu.u, st_gpu.u, atol, rtol)
    v_ok = @allowscalar cuNumeric.compare(st_cpu.v, st_gpu.v, atol, rtol)
    return (u_ok && v_ok) ? "pass" : "fail"
end

# VARIANT DESCRIPTION
# baseline: as written
# lifetimes: step wrapped in @analyze_lifetimes
let body = quote
        # currently we don't have NDArray^x working yet. every operator is dotted
        # so each rhs fuses into a single broadcast kernel rather than shattering
        # into bare +/-/* binary tasks.
        F_u = (
            (
                .-u[2:(end - 1), 2:(end - 1)] .*
                (v[2:(end - 1), 2:(end - 1)] .* v[2:(end - 1), 2:(end - 1)])
            ) .+ args.f .* (1.0f0 .- u[2:(end - 1), 2:(end - 1)])
        )
        F_v = (
            (
                u[2:(end - 1), 2:(end - 1)] .*
                (v[2:(end - 1), 2:(end - 1)] .* v[2:(end - 1), 2:(end - 1)])
            ) .- (args.f + args.k) .* v[2:(end - 1), 2:(end - 1)]
        )
        # 2-D Laplacian via slicing, excluding boundaries
        u_lap = (
            (
                u[3:end, 2:(end - 1)] .- 2 .* u[2:(end - 1), 2:(end - 1)] .+
                u[1:(end - 2), 2:(end - 1)]
            ) ./ args.dx^2 .+
            (
                u[2:(end - 1), 3:end] .- 2 .* u[2:(end - 1), 2:(end - 1)] .+
                u[2:(end - 1), 1:(end - 2)]
            ) ./ args.dx^2
        )
        v_lap = (
            (
                v[3:end, 2:(end - 1)] .- 2 .* v[2:(end - 1), 2:(end - 1)] .+
                v[1:(end - 2), 2:(end - 1)]
            ) ./ args.dx^2 .+
            (
                v[2:(end - 1), 3:end] .- 2 .* v[2:(end - 1), 2:(end - 1)] .+
                v[2:(end - 1), 1:(end - 2)]
            ) ./ args.dx^2
        )

        # Forward-Euler step for all interior points
        u_new[2:(end - 1), 2:(end - 1)] =
            ((args.c_u .* u_lap) .+ F_u) .* args.dt .+ u[2:(end - 1), 2:(end - 1)]
        v_new[2:(end - 1), 2:(end - 1)] =
            ((args.c_v .* v_lap) .+ F_v) .* args.dt .+ v[2:(end - 1), 2:(end - 1)]

        # Periodic boundary conditions
        u_new[:, 1] = u[:, end - 1]
        u_new[:, end] = u[:, 2]
        u_new[1, :] = u[end - 1, :]
        u_new[end, :] = u[2, :]
        v_new[:, 1] = v[:, end - 1]
        v_new[:, end] = v[:, 2]
        v_new[1, :] = v[end - 1, :]
        v_new[end, :] = v[2, :]
    end
    @eval _gs_step!(b::GrayScottBaseline, u, v, u_new, v_new, args::GSParams) = $body
    @eval _gs_step!(b::GrayScottLifetimes, u, v, u_new, v_new, args::GSParams) = @analyze_lifetimes $body
end

function run!(b::AbstractGrayScott, st::GrayScottState)
    _gs_step!(b, st.u, st.v, st.u_new, st.v_new, st.params)
    # swap references rather than copy
    st.u, st.u_new = st.u_new, st.u
    st.v, st.v_new = st.v_new, st.v
    return nothing
end

register_benchmark("grayscott_baseline", GrayScottBaseline)
register_benchmark("grayscott_lifetimes", GrayScottLifetimes)
