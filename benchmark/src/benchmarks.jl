using Printf
using Statistics

"""
- `n_warmup::Int` : Number of warmup steps. These are not timed. Intended
    to avoid pre-compilation cost being timed.
- `n_iter::Int` : Number of iterations to run per trial. Should be large enough
    to build up queue depth of tasks such that latency is hidden.
- `n_trial::Int` : Number of independent trials to run. Timing is restarted and
    legate in between each trial. Sets number of datapoints used to estimated
    standard deviations/errors.
- `n_gpu::Int` : The number of GPUs used by legate. Set through the LEGATE_CONFIG,
    this value is just bookkeeping.
"""
Base.@kwdef struct GlobalSettings
    n_warmup::Int # Number of warmup steps, where timing is not done.
    n_iter::Int # Number of iterations to run per trial
    n_trial::Int = 1 # Number of independent trials to run. Benchmark
    n_gpu::Int = 0
end

#########################################

abstract type AbstractBenchmark{T} end

#########################################

Base.@kwdef struct GEMM{T} <: AbstractBenchmark{T}
    N::Int
    M::Int
end

name(::GEMM) = "sgemm"
dims(g::GEMM) = (g.N, g.M)
data(g::GEMM{T}) where {T} = "GEMM with T=$(T), N=$(g.N), M=$(g.M)"

function allowed_types(::Type{GEMM})
    Union{cuNumeric.SUPPORTED_FLOAT_TYPES,cuNumeric.SUPPORTED_INT_TYPES}
end

total_flops(s::GEMM) = s.N * s.N * ((2*s.M) - 1)
total_space(s::GEMM{T}) where {T} = 2 * ((s.N*s.M) * sizeof(T)) + ((s.N*s.N) * sizeof(T))

function initialize(s::GEMM{T}) where {T}
    A = cuNumeric.rand(T, s.N, s.M)
    B = cuNumeric.rand(T, s.M, s.N)
    C = cuNumeric.zeros(T, s.N, s.N)
    GC.gc()
    return C, A, B
end

run!(::GEMM, C, A, B) = mul!(C, A, B)

#########################################

Base.@kwdef struct MonteCarloIntegration{T} <: AbstractBenchmark{T}
    n_samples::Int
end

name(::MonteCarloIntegration) = "montecarlo"
dims(mci::MonteCarloIntegration) = (mci.n_samples, 1)
function data(mci::MonteCarloIntegration{T}) where {T}
    "Monte Carlo Integration with T=$(T), n_samples=$(mci.n_samples)"
end

allowed_types(::Type{MonteCarloIntegration}) = cuNumeric.SUPPORTED_FLOAT_TYPES

total_space(s::MonteCarloIntegration{T}) where {T} = s.n_samples * sizeof(T)
total_flops(s::MonteCarloIntegration) = s.n_samples

function initialize(mci::MonteCarloIntegration{T}) where {T}
    # Uniform samples over the integration domain [0, 10].
    x = T(10) .* cuNumeric.rand(T, mci.n_samples)
    GC.gc()
    return (x,)
end

_domain_volume(mci::MonteCarloIntegration{T}) where {T} = T(10) / mci.n_samples
run!(mci::MonteCarloIntegration, x) = _domain_volume(mci) * sum(exp.(-x .^ 2))

#########################################

struct GSParams{T}
    dx::T
    dt::T
    c_u::T
    c_v::T
    f::T
    k::T
end

function GSParams{T}(; dx=1, c_u=1.0, c_v=0.3, f=0.03, k=0.06) where {T}
    GSParams{T}(T(dx), T(dx / 5), T(c_u), T(c_v), T(f), T(k))
end

Base.@kwdef struct GrayScott{T} <: AbstractBenchmark{T}
    N::Int
    M::Int
end

name(::GrayScott) = "grayscott"
dims(b::GrayScott) = (b.N, b.M)
data(b::GrayScott{T}) where {T} = "GrayScott with T=$(T), N=$(b.N), M=$(b.M)"
allowed_types(::Type{GrayScott}) = cuNumeric.SUPPORTED_FLOAT_TYPES
total_flops(b::GrayScott) = b.N * b.M # grid points updated per step

mutable struct GrayScottState{A,P}
    u::A
    v::A
    u_new::A
    v_new::A
    params::P
end

function initialize(b::GrayScott{T}) where {T}
    d = (b.N, b.M)
    u = cuNumeric.ones(T, d)
    v = cuNumeric.zeros(T, d)
    u_new = cuNumeric.zeros(T, d)
    v_new = cuNumeric.zeros(T, d)

    seed = min(150, b.N, b.M)
    u[1:seed, 1:seed] = cuNumeric.rand(T, (seed, seed))
    v[1:seed, 1:seed] = cuNumeric.rand(T, (seed, seed))

    return (GrayScottState(u, v, u_new, v_new, GSParams{T}()),)
end

function _gs_step!(u, v, u_new, v_new, args::GSParams)
    # currently we don't have NDArray^x working yet.
    F_u = (
        (
            -u[2:(end - 1), 2:(end - 1)] .*
            (v[2:(end - 1), 2:(end - 1)] .* v[2:(end - 1), 2:(end - 1)])
        ) + args.f * (1 .- u[2:(end - 1), 2:(end - 1)])
    )
    F_v = (
        (
            u[2:(end - 1), 2:(end - 1)] .*
            (v[2:(end - 1), 2:(end - 1)] .* v[2:(end - 1), 2:(end - 1)])
        ) - (args.f + args.k) * v[2:(end - 1), 2:(end - 1)]
    )
    # 2-D Laplacian via slicing, excluding boundaries
    u_lap = (
        (
            u[3:end, 2:(end - 1)] - 2 * u[2:(end - 1), 2:(end - 1)] +
            u[1:(end - 2), 2:(end - 1)]
        ) ./ args.dx^2 +
        (
            u[2:(end - 1), 3:end] - 2 * u[2:(end - 1), 2:(end - 1)] +
            u[2:(end - 1), 1:(end - 2)]
        ) ./ args.dx^2
    )
    v_lap = (
        (
            v[3:end, 2:(end - 1)] - 2 * v[2:(end - 1), 2:(end - 1)] +
            v[1:(end - 2), 2:(end - 1)]
        ) ./ args.dx^2 +
        (
            v[2:(end - 1), 3:end] - 2 * v[2:(end - 1), 2:(end - 1)] +
            v[2:(end - 1), 1:(end - 2)]
        ) ./ args.dx^2
    )

    # Forward-Euler step for all interior points
    u_new[2:(end - 1), 2:(end - 1)] =
        ((args.c_u * u_lap) + F_u) * args.dt + u[2:(end - 1), 2:(end - 1)]
    v_new[2:(end - 1), 2:(end - 1)] =
        ((args.c_v * v_lap) + F_v) * args.dt + v[2:(end - 1), 2:(end - 1)]

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

function run!(::GrayScott, st::GrayScottState)
    _gs_step!(st.u, st.v, st.u_new, st.v_new, st.params)
    # swap references rather than copy
    st.u, st.u_new = st.u_new, st.u
    st.v, st.v_new = st.v_new, st.v
    return nothing
end

#########################################

# Maps the benchmarks.toml table name to its benchmark type. Add new benchmarks here.
const BENCHMARKS = Dict{String,Type}(
    "sgemm" => GEMM,
    "grayscott" => GrayScott,
    "montecarlo" => MonteCarloIntegration,
)

# Construct a benchmark from the orchestrator's positional sizes. Most benchmarks
# use (N, M); MonteCarloIntegration uses N as its sample count and ignores M.
build_benchmark(::Type{B}, ::Type{T}, N, M) where {B<:AbstractBenchmark,T} = B{T}(; N=N, M=M)
function build_benchmark(::Type{MonteCarloIntegration}, ::Type{T}, N, M) where {T}
    MonteCarloIntegration{T}(; n_samples=N)
end

# Per-trial timings for one benchmark. `times_ms[i]`/`gflops[i]` are the mean
# over `n_iter` iterations for trial `i`; the spread across trials gives stddev.
struct BenchmarkResult{B<:AbstractBenchmark}
    times_ms::Vector{Float64}
    gflops::Vector{Float64}
    benchmark::B
end

# One timed trial: warmup, then time `n_iter` iterations of `run!`.
function _trial(b::AbstractBenchmark, gs::GlobalSettings)
    GC.gc(true)
    state = initialize(b)

    start_time = zero(get_time_microseconds())
    for idx in 1:(gs.n_warmup + gs.n_iter)
        if idx == gs.n_warmup + 1
            start_time = get_time_microseconds()
        end
        run!(b, state...)
    end
    total_time_μs = get_time_microseconds() - start_time

    mean_time_ms = total_time_μs / (gs.n_iter * 1e3)
    gflops = total_flops(b) / (mean_time_ms * 1e6)
    return mean_time_ms, gflops
end

# Run `n_trial` independent trials and collect their per-trial measurements.
function run_benchmark(b::AbstractBenchmark, gs::GlobalSettings)
    times_ms = Float64[]
    gflops = Float64[]
    for _ in 1:gs.n_trial
        t, g = _trial(b, gs)
        push!(times_ms, t)
        push!(gflops, g)
    end
    return BenchmarkResult(times_ms, gflops, b)
end

_std(x) = length(x) > 1 ? std(x) : 0.0

function save_result(br::BenchmarkResult, gpus)
    N, M = dims(br.benchmark)
    path = joinpath(@__DIR__, "..", "results", "$(name(br.benchmark)).csv")
    mkpath(dirname(path))
    open(path, "a") do io
        for trial in eachindex(br.times_ms)
            @printf(
                io, "%s,%d,%d,%d,%d,%.6f,%.6f\n",
                "cunumeric", gpus, N, M, trial, br.times_ms[trial], br.gflops[trial],
            )
        end
    end
end
