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
- `check_correctness::Bool` : If true, run one CPU-reference check per config
    (not per timed iteration) before timing; result is recorded in the CSV.
- `n_correctness_iter::Int` : Steps to run for that single correctness check.
"""
Base.@kwdef struct GlobalSettings
    n_warmup::Int # Number of warmup steps, where timing is not done.
    n_iter::Int # Number of iterations to run per trial
    n_trial::Int = 1 # Number of independent trials to run. Benchmark
    n_gpu::Int = 0
    cupynumeric::Bool = false # also run baselines under cupynumeric for comparison
    cuda::Bool = false # also run under CUDA.jl for comparison (single-GPU only)
    check_correctness::Bool = false
    n_correctness_iter::Int = 5
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

# Internal adapter for benchmark generators that share a quoted step body.
function _define_accelerated_definition(signature, body, form=:function)
    if form === :function
        return cuNumeric._accelerate_expand(Expr(:function, signature, body), @__MODULE__)
    end
    scoped = form === :begin ? Expr(:block, body.args...) : Expr(:let, body)
    call = Expr(:macrocall, Symbol("@accelerate"), LineNumberNode(0), scoped)
    return Expr(:function, signature, Expr(:block, Base.macroexpand(@__MODULE__, call)))
end

# Maps a benchmarks.toml table name to its benchmark type. Each benchmark file
# registers itself via `register_benchmark`.
const BENCHMARKS = Dict{String,Type}()
function register_benchmark(key::AbstractString, ::Type{B}) where {B<:AbstractBenchmark}
    return BENCHMARKS[key] = B
end

benchmark_backend_label(::AbstractBenchmark, backend::String, default::String) = default
benchmark_backend_save_as(::AbstractBenchmark, backend::String, default::String) = default

function build_benchmark(::Type{B}, ::Type{T}, N, M) where {B<:AbstractBenchmark,T}
    return B{T}(; N=N, M=M)
end

#########################################

# Per-trial timings for one benchmark. `times_ms[i]`/`gflops[i]` are the mean
# over `n_iter` iterations for trial `i`; the spread across trials gives stddev.
# `correctness` is one of "pass", "fail", "skipped" — checked once per config.
struct BenchmarkResult{B<:AbstractBenchmark}
    times_ms::Vector{Float64}
    gflops::Vector{Float64}
    benchmark::B
    correctness::String
end

# Optional per-benchmark correctness vs a CPU/`Array` reference.
# Return "pass", "fail", or "skipped". Default: no check implemented.
correctness_supported(::AbstractBenchmark) = false
function check_benchmark_correctness(b::AbstractBenchmark, gs::GlobalSettings; mod=cuNumeric)
    return "skipped"
end

# One timed trial: warmup, then time `n_iter` iterations of `run!`.
function _trial(
    b::AbstractBenchmark, gs::GlobalSettings;
    mod=cuNumeric, clock=get_time_microseconds,
)
    GC.gc(true)
    state = initialize(b; mod=mod)

    start_time = nothing
    for idx in 1:(gs.n_warmup + gs.n_iter)
        if idx == gs.n_warmup + 1
            start_time = clock()
        end
        run!(b, state...)
    end
    total_time_μs = clock() - start_time

    mean_time_ms = total_time_μs / (gs.n_iter * 1e3)
    gflops = total_flops(b) / (mean_time_ms * 1e6)
    return mean_time_ms, gflops
end

# Run `n_trial` independent trials and collect their per-trial measurements.
# Correctness (if enabled) runs once before timing, not per trial/iteration.
function run_benchmark(
    b::AbstractBenchmark, gs::GlobalSettings;
    mod=cuNumeric, clock=get_time_microseconds,
)
    correctness = "skipped"
    if gs.check_correctness
        if correctness_supported(b)
            correctness = check_benchmark_correctness(b, gs; mod=mod)
        else
            correctness = "skipped"
        end
    end

    times_ms = Float64[]
    gflops = Float64[]
    for _ in 1:gs.n_trial
        t, g = _trial(b, gs; mod=mod, clock=clock)
        push!(times_ms, t)
        push!(gflops, g)
    end
    return BenchmarkResult(times_ms, gflops, b, correctness)
end

_std(x) = length(x) > 1 ? std(x) : 0.0

function save_result(br::BenchmarkResult, gpus; mod::String="cunumeric")
    N, M = dims(br.benchmark)
    path = joinpath(@__DIR__, "..", "results", "$(name(br.benchmark))_$(mod).csv")
    mkpath(dirname(path))
    open(path, "a") do io
        for trial in eachindex(br.times_ms)
            # correctness is per-config; repeated on each trial row for CSV joins
            @printf(
                io, "%s,%d,%d,%d,%d,%.6f,%.6f,%s\n",
                mod, gpus, N, M, trial,
                br.times_ms[trial], br.gflops[trial], br.correctness,
            )
        end
    end
end

#########################################

# `setup` runs in the worker before the benchmark is built (e.g. flip a runtime
# preference); code-path variants leave it a no-op.
# struct Variant
#     name::String
#     setup::Function
# end

# const VARIANTS = Dict{String,Variant}()

# function register_variant(name, setup=() -> nothing)
#     VARIANTS[name] = Variant(name, setup)
# end

# function variant_setup(name)
#     if haskey(VARIANTS, name)
#         return VARIANTS[name].setup
#     end
#     return () -> nothing
# end

# register_variant("baseline")
# register_variant("fusion_off", cuNumeric.disable_broadcast_fusion!)
# register_variant("fusion_on",  cuNumeric.enable_broadcast_fusion!)
