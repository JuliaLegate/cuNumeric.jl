# Dedicated CUDA.jl baseline worker. This environment contains none of the
# cuNumeric, JACC, or Dagger packages.
include(joinpath(@__DIR__, "..", "model_isolation.jl"))
assert_active_model(:cudajl)

using CUDA
using AbstractFFTs
using LinearAlgebra
using TensorOperations
using cuTENSOR

assert_models_not_loaded(("cuNumeric", "JACC", "Dagger"))

function array_backend_entry()
    cuda_sync() = CUDA.synchronize(; blocking=true)
    cuda_clock() = (cuda_sync(); time_ns()/1e3)
    return (
        id=:cudajl, mod=CUDA, label="CUDA.jl", save_as="CUDA.jl",
        clock=cuda_clock, synchronize=cuda_sync, fused=()->true,
    )
end

include(joinpath(@__DIR__, "..", "core.jl"))
include_benchmarks()
include(joinpath(@__DIR__, "benchmarks", "montecarlo.jl"))
include(joinpath(@__DIR__, "..", "array_worker.jl"))
