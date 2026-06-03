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

# Interface each benchmark implements (see benchmarks/gemm.jl for a template).
function name end
function dims end
function data end
function allowed_types end
function total_flops end
function initialize end
function run! end

# Maps a benchmarks.toml table name to its benchmark type. Each benchmark file
# registers itself via `register_benchmark`.
const BENCHMARKS = Dict{String,Type}()
function register_benchmark(key::AbstractString, ::Type{B}) where {B<:AbstractBenchmark}
    BENCHMARKS[key] = B
end

# Default uses (N, M); benchmarks with a code-path variant or different arity
# override this (see grayscott.jl / montecarlo.jl).
function build_benchmark(::Type{B}, ::Type{T}, N, M, variant) where {B<:AbstractBenchmark,T}
    B{T}(; N=N, M=M)
end

#########################################

# `setup` runs in the worker before the benchmark is built (e.g. flip a runtime
# preference); code-path variants leave it a no-op.
struct Variant
    name::String
    setup::Function
end

const VARIANTS = Dict{String,Variant}()

function register_variant(name, setup=() -> nothing)
    VARIANTS[name] = Variant(name, setup)
end

function variant_setup(name)
    if haskey(VARIANTS, name)
        return VARIANTS[name].setup
    end
    return () -> nothing
end

register_variant("baseline")
# register_variant("fusion_off", cuNumeric.CNPreferences.disable_broadcast_fusion!)
# register_variant("fusion_on",  cuNumeric.CNPreferences.enable_broadcast_fusion!)

#########################################

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

function save_result(br::BenchmarkResult, gpus, variant)
    N, M = dims(br.benchmark)
    path = joinpath(@__DIR__, "..", "results", "$(name(br.benchmark)).csv")
    mkpath(dirname(path))
    open(path, "a") do io
        for trial in eachindex(br.times_ms)
            @printf(
                io, "%s,%s,%d,%d,%d,%d,%.6f,%.6f\n",
                "cunumeric", variant, gpus, N, M, trial,
                br.times_ms[trial], br.gflops[trial],
            )
        end
    end
end
