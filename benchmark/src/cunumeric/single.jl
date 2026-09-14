# Dedicated cuNumeric worker. CUDA.jl is loaded only for the optional tiny
# single-GPU correctness oracle; JACC and Dagger are never loaded here.
include(joinpath(@__DIR__, "..", "model_isolation.jl"))
assert_active_model(:cunumeric)

using cuNumeric
using LinearAlgebra
using TensorOperations

const NEED_CUDA_ORACLE =
    length(ARGS) >= 9 && parse(Int,ARGS[1]) == 1 && parse(Bool,ARGS[9])
if NEED_CUDA_ORACLE
    using CUDA
    using AbstractFFTs
    using cuTENSOR
end

assert_models_not_loaded(("JACC", "Dagger"))

function array_backend_entry()
    return (
        id=:cunumeric,mod=cuNumeric,label="cuNumeric",save_as="cunumeric",
        clock=get_time_microseconds,
        synchronize=benchmark_synchronize,
        fused=()->cuNumeric.FUSE_BROADCAST_EXPRS,
    )
end

include(joinpath(@__DIR__, "..", "array_worker.jl"))
