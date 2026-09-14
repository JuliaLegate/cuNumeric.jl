# Dedicated Dagger worker. CUDA is Dagger's device backend in this environment;
# this process never imports cuNumeric or JACC.
include(joinpath(@__DIR__, "..", "model_worker.jl"))
assert_active_model(:dagger)

using CUDA: CUDA
using Dagger: Dagger

assert_models_not_loaded(("cuNumeric", "JACC"))
include(joinpath(@__DIR__, "benchmarks", "montecarlo.jl"))

run_model_worker(:dagger, "Dagger.jl")
