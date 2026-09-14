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

run_model_worker(:jacc, "JACC.jl")
