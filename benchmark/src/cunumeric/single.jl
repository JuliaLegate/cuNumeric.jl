# Dedicated cuNumeric worker. JACC and Dagger are never loaded here.
include(joinpath(@__DIR__, "..", "model_isolation.jl"))
assert_active_model(:cunumeric)

using cuNumeric
using AbstractFFTs
using LinearAlgebra, TOML
using TensorOperations

assert_models_not_loaded(("JACC", "Dagger"))

function array_backend_entry()
    return (
        id=:cunumeric, mod=cuNumeric, label="cuNumeric", save_as="cunumeric",
        clock=get_time_microseconds,
        synchronize=benchmark_synchronize,
        fused=()->cuNumeric.FUSE_BROADCAST_EXPRS,
    )
end

include(joinpath(@__DIR__, "..", "core.jl"))
include_benchmarks()
include(joinpath(@__DIR__, "benchmarks", "cg.jl"))
include(joinpath(@__DIR__, "benchmarks", "nas", "ep.jl"))
include(joinpath(@__DIR__, "benchmarks", "nas", "ft.jl"))
include(joinpath(@__DIR__, "benchmarks", "montecarlo.jl"))
include(joinpath(@__DIR__, "benchmarks", "grayscott_accelerate_forms.jl"))

function needs_cuda_correctness_oracle(args)
    length(args) >= 9 || return false
    parse(Int, args[1]) == 1 && parse(Bool, args[9]) || return false
    name = args[2]
    haskey(BENCHMARKS, name) || return false
    T = get(Dict("Float32"=>Float32, "Float64"=>Float64), args[3], nothing)
    T === nothing && return false
    kwargs = length(args)==12 ? Dict(Symbol(k)=>v for (k, v) in TOML.parse(args[12])) : Dict()
    benchmark = build_benchmark(
        BENCHMARKS[name], T, parse(Int, args[4]), parse(Int, args[5]); kwargs...
    )
    return !correctness_uses_cpu(benchmark)
end

const NEED_CUDA_ORACLE = needs_cuda_correctness_oracle(ARGS)
if NEED_CUDA_ORACLE
    using CUDA
    using AbstractFFTs
    using cuTENSOR
end

include(joinpath(@__DIR__, "..", "array_worker.jl"))
