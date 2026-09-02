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

Base.@kwdef struct GrayScottAccelerated{T} <: AbstractGrayScott{T}
    N::Int
    M::Int
end

name(::GrayScottBaseline) = "grayscott_baseline"
name(::GrayScottAccelerated) = "grayscott_accelerated"
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
    u = ones_array(mod, T, b.N, b.M)
    v = zeros_array(mod, T, b.N, b.M)
    u_new = zeros_array(mod, T, b.N, b.M)
    v_new = zeros_array(mod, T, b.N, b.M)

    seed = min(150, b.N, b.M)
    if deterministic
        # Shared host pattern so cuNumeric and CUDA.jl start from the same IC.
        host_u = T[T(0.5) + T(0.5) * sin(T(i)) * cos(T(j)) for i in 1:seed, j in 1:seed]
        host_v = T[T(0.25) + T(0.25) * cos(T(i)) * sin(T(j)) for i in 1:seed, j in 1:seed]
        u[1:seed, 1:seed] = to_backend(mod, host_u)
        v[1:seed, 1:seed] = to_backend(mod, host_v)
    else
        u[1:seed, 1:seed] = rand_array(mod, T, seed, seed)
        v[1:seed, 1:seed] = rand_array(mod, T, seed, seed)
    end

    return (GrayScottState(u, v, u_new, v_new, GSParams{T}()),)
end

function to_backend_state(mod, st::GrayScottState)
    return GrayScottState(
        to_backend_state(mod, st.u),
        to_backend_state(mod, st.v),
        to_backend_state(mod, st.u_new),
        to_backend_state(mod, st.v_new),
        st.params,
    )
end

function correctness_problem(b::AbstractGrayScott{T}) where {T}
    return typeof(b)(; N=min(32, b.N, b.M), M=min(32, b.N, b.M))
end
correctness_iters(::AbstractGrayScott, gs::GlobalSettings) = gs.n_correctness_iter
correctness_result(::AbstractGrayScott, state, _) = (only(state).u, only(state).v)
function cuda_runnable(b::GrayScottAccelerated{T}) where {T}
    return GrayScottBaseline{T}(; N=b.N, M=b.M)
end

# Shared syntax tree keeps every Gray-Scott variant on the exact same workload.
const GRAYSCOTT_STEP_BODY = quote
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

# Original baseline and recommended function-form benchmark.
let body = deepcopy(GRAYSCOTT_STEP_BODY)
    @eval _gs_step!(b::GrayScottBaseline, u, v, u_new, v_new, args::GSParams) = $body
    definition = _define_accelerated_definition(
        :(_gs_step!(b::GrayScottAccelerated, u, v, u_new, v_new, args::GSParams)), body
    )
    @eval $definition
end

function run!(b::AbstractGrayScott, st::GrayScottState)
    _gs_step!(b, st.u, st.v, st.u_new, st.v_new, st.params)
    # swap references rather than copy
    st.u, st.u_new = st.u_new, st.u
    st.v, st.v_new = st.v_new, st.v
    return nothing
end

register_benchmark("grayscott_baseline", GrayScottBaseline)
register_benchmark("grayscott_accelerated", GrayScottAccelerated)
