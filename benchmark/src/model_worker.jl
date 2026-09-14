# Shared worker protocol for execution models with native benchmark programs.
# Model packages are imported by their entrypoint before `run_model_worker` is
# called; this file never imports an execution model itself.

using Printf
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
        parse(Int, args[1]),args[2],T,T_name,parse(Int,args[4]),parse(Int,args[5]),
        parse(Int,args[6]),parse(Int,args[7]),parse(Int,args[8]),parse(Bool,args[9]),
        parse(Int,args[10]),parse(Float64,args[11]),
    )
    config.gpus > 0 || error("gpus must be positive")
    config.n_iter > 0 && config.n_trial > 0 && config.n_warmup >= 0 ||
        error("invalid trial/iteration count")
    return config
end

model_fence_each_iteration(benchmark) = true
model_synchronize(benchmark) = nothing

function model_trial(benchmark, config)
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

    start = time_ns()
    for _ in 1:config.n_iter
        model_run!(benchmark, state)
        fence_each && model_synchronize(benchmark)
    end
    model_synchronize(benchmark)
    elapsed_us = (time_ns() - start) / 1e3

    mean_time_ms = elapsed_us / (config.n_iter * 1e3)
    gflops = config.flops / (mean_time_ms * 1e6)
    return mean_time_ms, gflops
end

function save_model_results(config, model::Symbol, times_ms, gflops)
    results = get(ENV, "CUNUMERIC_BENCH_RESULTS_DIR", joinpath(@__DIR__, "..", "results"))
    path = joinpath(results, "$(config.name)_$(model).csv")
    mkpath(dirname(path))
    open(path, "a") do io
        for trial in eachindex(times_ms)
            @printf(
                io,"%s,%d,%d,%d,%d,%.6f,%.6f,skipped\n",
                model,config.gpus,config.N,config.M,trial,times_ms[trial],gflops[trial],
            )
        end
    end
    return path
end

function run_model_worker(model::Symbol, label::String, args=ARGS)
    assert_active_model(model)
    config = parse_model_worker_args(args)
    benchmark = model_build_benchmark(config)
    config.check_correctness && @warn(
        "Correctness checking is not implemented for $label; recording skipped",
    )
    println(
        "[$label] $(config.name) benchmark ($(config.T_name)) on " *
        "$(config.N)x$(config.M) for $(config.n_iter) iterations " *
        "($(config.n_warmup) warmup) x $(config.n_trial) trials",
    )

    times_ms = Float64[]
    gflops = Float64[]
    for trial in 1:config.n_trial
        time_ms, throughput = model_trial(benchmark, config)
        push!(times_ms, time_ms)
        push!(gflops, throughput)
        @printf(
            "[%s] Trial %d/%d: %.5f ms, %.5f GFLOPS\n",
            label,trial,config.n_trial,time_ms,throughput,
        )
    end
    @printf("[%s] Mean Run Time: %.5f ± %.5f ms\n",label,mean(times_ms),length(times_ms)>1 ? std(times_ms) : 0.0)
    @printf("[%s] FLOPS: %.5f ± %.5f GFLOPS\n",label,mean(gflops),length(gflops)>1 ? std(gflops) : 0.0)
    println("[$label] Correctness: skipped")
    return save_model_results(config,model,times_ms,gflops)
end
