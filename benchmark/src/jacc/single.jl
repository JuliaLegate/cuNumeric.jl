# Dedicated JACC worker. This process never imports cuNumeric or Dagger.
include(joinpath(@__DIR__, "..", "model_worker.jl"))
assert_active_model(:jacc)

import JACC
JACC.@init_backend

assert_models_not_loaded(("cuNumeric", "Dagger"))
include(joinpath(@__DIR__, "benchmarks", "montecarlo.jl"))

run_model_worker(:jacc, "JACC.jl")
