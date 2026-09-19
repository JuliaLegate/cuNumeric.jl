# Dedicated Dagger worker. CUDA is Dagger's device backend in this environment;
# this process never imports cuNumeric or JACC.
include(joinpath(@__DIR__, "..", "model_worker.jl"))
assert_active_model(:dagger)

using CUDA: CUDA
using Dagger: Dagger
using AbstractFFTs
import Dagger: @stencil, Pad, Wrap
using LinearAlgebra

assert_models_not_loaded(("cuNumeric", "JACC"))
include(joinpath(@__DIR__, "common.jl"))
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
        "Dagger benchmark '$(config.name)' is not implemented; known: " *
        join(SUPPORTED_BENCHMARKS, ", "),
    )

    if config.name == "nas_ep"
        return model_build_nas_ep(config)
    elseif config.name == "nas_ft"
        return model_build_nas_ft(config)
    elseif config.name == "nas_mg"
        return model_build_nas_mg(config)
    elseif config.name == "montecarlo"
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
