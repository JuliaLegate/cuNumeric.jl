# Dedicated JACC worker. This process never imports cuNumeric or Dagger.
include(joinpath(@__DIR__, "..", "model_worker.jl"))
assert_active_model(:jacc)

using Logging
get(ENV, "CUNUMERIC_BENCH_VERBOSE", "0") == "1" ||
    global_logger(ConsoleLogger(stderr, Logging.Warn))
using JACC: JACC
JACC.@init_backend
using CUDA
using AbstractFFTs

assert_models_not_loaded(("cuNumeric", "Dagger"))
include(joinpath(@__DIR__, "benchmarks", "montecarlo.jl"))
include(joinpath(@__DIR__, "benchmarks", "gemm.jl"))
include(joinpath(@__DIR__, "benchmarks", "grayscott.jl"))

include(joinpath(@__DIR__, "benchmarks", "cg.jl"))
include(joinpath(@__DIR__, "benchmarks", "nas", "ep.jl"))
include(joinpath(@__DIR__, "benchmarks", "nas", "ft.jl"))
include(joinpath(@__DIR__, "benchmarks", "nas", "mg.jl"))

const SUPPORTED_BENCHMARKS = [
    "montecarlo", "gemm", "grayscott", "cg", "nas_ep", "nas_ft", "nas_mg"
]

function model_build_benchmark(config::ModelWorkerConfig)
    config.name in SUPPORTED_BENCHMARKS || error(
        "JACC benchmark '$(config.name)' is not implemented; known: " *
        join(SUPPORTED_BENCHMARKS, ", "),
    )

    if config.name == "nas_ep"
        return model_build_nas_ep(config)
    elseif config.name == "nas_ft"
        return model_build_nas_ft(config)
    elseif config.name == "nas_mg"
        return model_build_nas_mg(config)
    elseif config.name == "cg"
        return model_build_cg(config)
    elseif config.name == "montecarlo"
        return model_build_montecarlo(config)
    elseif config.name == "gemm"
        return model_build_gemm(config)
    elseif config.name == "grayscott"
        return model_build_grayscott(config)
    end
end

run_model_worker(:jacc, "JACC.jl")
