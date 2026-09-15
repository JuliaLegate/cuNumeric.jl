# Shared worker protocol for execution models with native benchmark programs.
# Model packages are imported by their entrypoint before `run_model_worker` is
# called; this file never imports an execution model itself.

using Printf
using ProgressMeter: ProgressMeter
using Statistics

include(joinpath(@__DIR__, "model_isolation.jl"))

struct ModelWorkerConfig
    gpus::Int
    name::String
    T::DataType
    T_name::String
    N::Int
    M::Int
    n_iter::Int
    n_warmup::Int
    n_trial::Int
    check_correctness::Bool
    n_correctness_iter::Int
    flops::Float64
end

function parse_model_worker_args(args)
    length(args) == 11 || error(
        "worker args: <gpus> <name> <T> <N> <M> <n_iter> <n_warmup> " *
        "<n_trial> <check_correctness> <n_correctness_iter> <flops>",
    )
    T_name = args[3]
    T = get(Dict("Float32" => Float32, "Float64" => Float64), T_name, nothing)
    T === nothing && error("Unsupported element type '$T_name'; known: Float32, Float64")
    config = ModelWorkerConfig(
        parse(Int, args[1]), args[2], T, T_name, parse(Int, args[4]), parse(Int, args[5]),
        parse(Int, args[6]), parse(Int, args[7]), parse(Int, args[8]), parse(Bool, args[9]),
        parse(Int, args[10]), parse(Float64, args[11]),
    )
    config.gpus > 0 || error("gpus must be positive")
    config.n_iter > 0 && config.n_trial > 0 && config.n_warmup >= 0 ||
        error("invalid trial/iteration count")
    return config
end

model_fence_each_iteration(benchmark) = true
model_synchronize(benchmark) = nothing
model_check_correctness(benchmark, config) = "skipped"
model_correctness_context(benchmark, config) = nothing

function montecarlo_correctness_samples(::Type{T}, n::Integer) where {T}
    return T.(range(T(0), T(10); length=n))
end

function montecarlo_correctness_reference(samples::AbstractVector{T}) where {T}
    return (T(10) / length(samples)) * sum(x -> exp(-(x*x)), samples)
end

function montecarlo_correctness_status(actual, expected, ::Type{T}) where {T}
    tolerance = T <: Float32 ? 1.0f-3 : 1e-10
    return isapprox(actual, expected; atol=tolerance, rtol=tolerance) ? "pass" : "fail"
end

# Shared host Gray-Scott reference (fully-periodic forward-Euler) for JACC/Dagger.
function grayscott_gs_params(::Type{T}) where {T}
    dx = T(1)
    return (dt=T(dx / 5), dx2=dx * dx, cu=T(1.0), cv=T(0.3), f=T(0.03), k=T(0.06))
end

function grayscott_host_init(::Type{T}, N, M; deterministic=false) where {T}
    u = ones(T, N, M)
    v = zeros(T, N, M)
    seed = min(150, N, M)
    if deterministic
        for j in 1:seed, i in 1:seed
            u[i, j] = T(0.5) + T(0.5) * sin(T(i)) * cos(T(j))
            v[i, j] = T(0.25) + T(0.25) * cos(T(i)) * sin(T(j))
        end
    else
        u[1:seed, 1:seed] = rand(T, seed, seed)
        v[1:seed, 1:seed] = rand(T, seed, seed)
    end
    return u, v
end

function grayscott_cpu_steps(::Type{T}, u0, v0, steps, p) where {T}
    N, M = size(u0)
    u, v = copy(u0), copy(v0)
    un, vn = similar(u), similar(v)
    for _ in 1:steps
        @inbounds for j in 1:M, i in 1:N
            up, vp = u[i, j], v[i, j]
            im = i == 1 ? N : i - 1
            ip = i == N ? 1 : i + 1
            jm = j == 1 ? M : j - 1
            jp = j == M ? 1 : j + 1
            lu = (u[ip, j] - 2up + u[im, j]) / p.dx2 + (u[i, jp] - 2up + u[i, jm]) / p.dx2
            lv = (v[ip, j] - 2vp + v[im, j]) / p.dx2 + (v[i, jp] - 2vp + v[i, jm]) / p.dx2
            uvv = up * vp * vp
            un[i, j] = up + p.dt * (p.cu * lu - uvv + p.f * (one(T) - up))
            vn[i, j] = vp + p.dt * (p.cv * lv + uvv - (p.f + p.k) * vp)
        end
        u, un = un, u
        v, vn = vn, v
    end
    return u, v
end

function grayscott_correctness_status(gu, gv, cu, cv, ::Type{T}) where {T}
    tol = T <: Float32 ? 1.0f-3 : 1e-10
    ok = isapprox(gu, cu; atol=tol, rtol=tol) && isapprox(gv, cv; atol=tol, rtol=tol)
    return ok ? "pass" : "fail"
end

function model_trial(benchmark, config; clock=time_ns)
    GC.gc(true)
    state = model_initialize(benchmark)
    fence_each = model_fence_each_iteration(benchmark)

    for _ in 1:config.n_warmup
        model_run!(benchmark, state)
        fence_each && model_synchronize(benchmark)
    end
    # Initialization and warmup work must be complete before starting the CPU
    # clock, including for models whose operations build asynchronous graphs.
    model_synchronize(benchmark)

    start = clock()
    for _ in 1:config.n_iter
        model_run!(benchmark, state)
        fence_each && model_synchronize(benchmark)
    end
    model_synchronize(benchmark)
    elapsed_us = (clock() - start) / 1e3

    mean_time_ms = elapsed_us / (config.n_iter * 1e3)
    gflops = config.flops / (mean_time_ms * 1e6)
    return mean_time_ms, gflops
end

function save_model_results(config, model::Symbol, times_ms, gflops, correctness)
    results = get(ENV, "CUNUMERIC_BENCH_RESULTS_DIR", joinpath(@__DIR__, "..", "results"))
    path = joinpath(results, "$(config.name)_$(model).csv")
    mkpath(dirname(path))
    open(path, "a") do io
        for trial in eachindex(times_ms)
            @printf(
                io, "%s,%d,%d,%d,%d,%.6f,%.6f,%s\n",
                model, config.gpus, config.N, config.M, trial, times_ms[trial], gflops[trial],
                correctness,
            )
        end
    end
    return path
end

function run_model_worker(model::Symbol, label::String, args=ARGS)
    assert_active_model(model)
    config = parse_model_worker_args(args)
    benchmark = model_build_benchmark(config)
    verbose = get(ENV, "CUNUMERIC_BENCH_VERBOSE", "0") == "1"
    if verbose && config.check_correctness
        context = model_correctness_context(benchmark, config)
        context !== nothing && println(
            "Correctness check: reference=$(context.reference), " *
            "dimensions=$(join(context.dims, '×'))",
        )
    end
    correctness = config.check_correctness ? model_check_correctness(benchmark, config) : "skipped"
    verbose && println(
        "[$label] trials=$(config.n_trial), warmups=$(config.n_warmup), " *
        "iterations=$(config.n_iter)",
    )

    times_ms = Float64[]
    gflops = Float64[]
    progress = ProgressMeter.Progress(
        config.n_trial; dt=0.0, desc="$(config.name) trials: ", barlen=40
    )
    ProgressMeter.update!(progress, 0)
    for trial in 1:config.n_trial
        time_ms, throughput = model_trial(benchmark, config)
        push!(times_ms, time_ms)
        push!(gflops, throughput)
        ProgressMeter.next!(
            progress;
            showvalues=[
                ("Completed trials", "$(trial)/$(config.n_trial)"),
                ("Last trial mean (ms/iteration)", @sprintf("%.5f", time_ms)),
                ("Last trial GFLOP/s", @sprintf("%.5f", throughput)),
            ],
        )
    end
    println("[$label] Correctness: $correctness")
    @printf(
        "[%s] Mean time: %.5f ± %.5f ms (trial SD)\n",
        label, mean(times_ms), length(times_ms)>1 ? std(times_ms) : 0.0,
    )
    @printf(
        "[%s] Mean throughput: %.5f ± %.5f GFLOP/s (trial SD)\n",
        label, mean(gflops), length(gflops)>1 ? std(gflops) : 0.0,
    )
    return save_model_results(config, model, times_ms, gflops, correctness)
end
