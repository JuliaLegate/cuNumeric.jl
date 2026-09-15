# Dedicated JACC worker. This process never imports cuNumeric or Dagger.
include(joinpath(@__DIR__, "..", "model_worker.jl"))
assert_active_model(:jacc)

using Logging
get(ENV, "CUNUMERIC_BENCH_VERBOSE", "0") == "1" ||
    global_logger(ConsoleLogger(stderr, Logging.Warn))
using JACC: JACC
JACC.@init_backend

assert_models_not_loaded(("cuNumeric", "Dagger"))
include(joinpath(@__DIR__, "benchmarks", "montecarlo.jl"))
include(joinpath(@__DIR__, "benchmarks", "gemm.jl"))

const SUPPORTED_BENCHMARKS = ["montecarlo", "gemm"]

function model_build_benchmark(config::ModelWorkerConfig)
    config.name in SUPPORTED_BENCHMARKS || error(
        "JACC benchmark '$(config.name)' is not implemented; known: " *
        join(SUPPORTED_BENCHMARKS, ", "),
    )

    if config.name == "montecarlo"
        return model_build_montecarlo(config)
    elseif config.name == "gemm"
        return model_build_gemm(config)
    end
end

run_model_worker(:jacc, "JACC.jl")
