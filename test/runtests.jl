using cuNumeric: cuNumeric
using CUDA: CUDA
using ParallelTestRunner
using Pkg
using InteractiveUtils: versioninfo

run_gpu_tests = CUDA.functional()

@info "Julia information:\n" * sprint(io -> versioninfo(io))
run_gpu_tests && @info "CUDA information:\n" * sprint(io -> CUDA.versioninfo(io))
@info "cuNumeric information:\n" * sprint(io -> cuNumeric.versioninfo(io))

########################################################
########################################################

# Forcibly precompile the current environment in parallel: Pkg sometimes ignores
# dependencies pointed through via `[sources]`
Pkg.precompile()

cuda_init = if run_gpu_tests
    quote
        using CUDA
        import CUDA: i32
    end
else
    :()
end

const init_code = quote
    using LinearAlgebra
    using Random
    import Random: rand

    $cuda_init

    include("util.jl")
end

testsuite = find_tests(@__DIR__)
delete!(testsuite, "util")

run_fusion_tests = run_gpu_tests && cuNumeric.FUSE_BROADCAST_EXPRS

if !run_fusion_tests
    @warn "Fusion tests will not be run. Either CUDA is not available or fusion is disabled."
    delete!(testsuite, "tests/broadcast_fusion")
end

runtests(cuNumeric, ARGS; testsuite, init_code)
