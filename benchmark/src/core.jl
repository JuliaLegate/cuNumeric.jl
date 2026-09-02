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
- `check_correctness::Bool` : If true and `n_gpu == 1`, compare a tiny cuNumeric
    result against CUDA.jl before timing. CUDA.jl / multi-GPU / Python skip.
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

# Optional hooks for the generic CUDA.jl check (initialize + run!).
correctness_problem(b::AbstractBenchmark) = b
correctness_iters(::AbstractBenchmark, gs::GlobalSettings) = 1
cuda_runnable(b::AbstractBenchmark) = b
correctness_result(::AbstractBenchmark, state, out) = out === nothing ? state : out
correctness_atol_rtol(::AbstractBenchmark, ::Type{T}) where {T} = ref_atol_rtol(T)

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

# CUDA.jl 6: the worker may pass `CUDA` or `CUDACore` as `mod`.
is_cuda_backend(mod) = nameof(mod) === :CUDA || nameof(mod) === :CUDACore

# Timed CUDA.jl is never the thing we check. Oracle compare is cuNumeric vs CUDA
# on a single GPU (the cuNumeric worker loads CUDA for the tiny problem).
function correctness_applies(gs::GlobalSettings, mod)
    is_cuda_backend(mod) && return false
    return gs.n_gpu == 1
end

function cuda_backend()
    for (id, mod) in Base.loaded_modules
        id.name == "CUDA" && return mod
    end
    return error("CUDA.jl must be loaded to check cuNumeric against CUDA.jl")
end

# 1–4D (or more) constructors. `mod` is cuNumeric or CUDA; both expose
# rand/zeros/ones(::Type, dims...). Host `Array` is only a seed for to_backend.
rand_array(mod, ::Type{T}, dims::Integer...) where {T} = mod.rand(T, dims...)
rand_array(mod, ::Type{T}, dims::Tuple) where {T} = rand_array(mod, T, dims...)
zeros_array(mod, ::Type{T}, dims::Integer...) where {T} = mod.zeros(T, dims...)
zeros_array(mod, ::Type{T}, dims::Tuple) where {T} = zeros_array(mod, T, dims...)
ones_array(mod, ::Type{T}, dims::Integer...) where {T} = mod.ones(T, dims...)
ones_array(mod, ::Type{T}, dims::Tuple) where {T} = ones_array(mod, T, dims...)

# Avoid `::NDArray` in the signature so the orchestrator can include this file
# without loading cuNumeric.
function astype_array(A, ::Type{T}) where {T}
    nameof(typeof(A)) === :NDArray && return cuNumeric.as_type(A, T)
    return T.(A)
end

to_host(A::Array) = A
to_host(x::Number) = x
to_host(A) = Array(A)

function to_backend(mod, A::AbstractArray)
    h = A isa Array ? A : Array(A)
    mod === cuNumeric && return NDArray(h)
    is_cuda_backend(mod) && return mod.CuArray(h)
    return h
end

to_backend_state(mod, x::AbstractArray) = to_backend(mod, x)
to_backend_state(mod, x::Tuple) = map(s -> to_backend_state(mod, s), x)
to_backend_state(mod, x) = x

function ref_atol_rtol(::Type{T}; atol=nothing, rtol=nothing) where {T}
    default = T <: Float32 ? 1.0f-3 : 1e-10
    return something(atol, default), something(rtol, default)
end

function isapprox_ref(actual, expected, ::Type{T}; atol=nothing, rtol=nothing) where {T}
    at, rt = ref_atol_rtol(T; atol, rtol)
    return isapprox(to_host(actual), to_host(expected); atol=at, rtol=rt)
end

_all_approx(a, b, ::Type{T}; kwargs...) where {T} = isapprox_ref(a, b, T; kwargs...)
function _all_approx(a::Tuple, b::Tuple, ::Type{T}; kwargs...) where {T}
    length(a) == length(b) || return false
    return all(_all_approx(x, y, T; kwargs...) for (x, y) in zip(a, b))
end

# Host-seed once, upload to both backends, initialize/run! the same way as timing.
function check_benchmark_correctness(
    b::AbstractBenchmark{T}, gs::GlobalSettings; mod=cuNumeric
) where {T}
    tiny = correctness_problem(b)
    seed = initialize(tiny; mod=Base)
    atol, rtol = correctness_atol_rtol(b, T)
    nstep = correctness_iters(tiny, gs)
    return check_vs_cuda(T; atol, rtol) do backend
        kernel = backend === cuNumeric ? tiny : cuda_runnable(tiny)
        state = to_backend_state(backend, seed)
        out = nothing
        for _ in 1:nstep
            out = run!(kernel, state...)
        end
        return correctness_result(kernel, state, out)
    end
end

# `f(mod)` runs the tiny problem on one backend and returns the value(s) to compare.
function check_vs_cuda(f, ::Type{T}; atol=nothing, rtol=nothing) where {T}
    got = f(cuNumeric)
    ref = f(cuda_backend())
    return _all_approx(got, ref, T; atol, rtol) ? "pass" : "fail"
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
        if correctness_applies(gs, mod)
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
