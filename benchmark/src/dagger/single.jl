# Dedicated Dagger worker. CUDA is Dagger's device backend in this environment;
# this process never imports cuNumeric or JACC.
include(joinpath(@__DIR__, "..", "model_worker.jl"))
assert_active_model(:dagger)

using CUDA: CUDA
using Dagger: Dagger
import Dagger: @stencil, Pad, Wrap
using LinearAlgebra

assert_models_not_loaded(("cuNumeric", "JACC"))
include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "benchmarks", "montecarlo.jl"))
include(joinpath(@__DIR__, "benchmarks", "gemm.jl"))
include(joinpath(@__DIR__, "benchmarks", "grayscott.jl"))
include(joinpath(@__DIR__, "benchmarks", "cg.jl"))

const SUPPORTED_BENCHMARKS = ["montecarlo", "gemm", "grayscott", "cg"]

function model_build_benchmark(config::ModelWorkerConfig)
    config.name in SUPPORTED_BENCHMARKS || error(
        "Dagger benchmark '$(config.name)' is not implemented; known: " *
        join(SUPPORTED_BENCHMARKS, ", "),
    )

    if config.name == "montecarlo"
        return model_build_montecarlo(config)
    elseif config.name == "gemm"
        return model_build_gemm(config)
    elseif config.name == "grayscott"
        return model_build_grayscott(config)
    elseif config.name == "cg"
        return model_build_cg(config)
    end
end

run_model_worker(:dagger, "Dagger.jl")
