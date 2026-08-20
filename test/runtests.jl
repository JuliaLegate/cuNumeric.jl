using cuNumeric
using CUDA: CUDA
using ParallelTestRunner
using Pkg
using InteractiveUtils: versioninfo

run_gpu_tests = CUDA.functional()

@info "Julia information:\n" * sprint(io -> versioninfo(io))
@info "cuNumeric information:\n" * sprint(io -> cuNumeric.versioninfo(io))

# Forcibly precompile the current environment in parallel: Pkg sometimes ignores
# dependencies pointed through via `[sources]`
Pkg.precompile()

const init_code = quote
    using LinearAlgebra
    using Random
    using StatsBase
    using FFTW
    import Random: rand

    ENV["LEGATE_SKIP_RUNTIME"] = "false"
    ENV["CUNUMERIC_PARALLEL_TEST_RUNNER"] = "1"
    using cuNumeric

    include("util.jl")
end

# Find all tests, remove ones that are not relevant to the current configuration

testsuite = find_tests(@__DIR__)
delete!(testsuite, "util")
delete!(testsuite, "array/unary/tests")
delete!(testsuite, "array/binary/tests")

test_args = parse_args(ARGS)
if filter_tests!(testsuite, test_args)
    if !run_gpu_tests
        @warn "CUDA GPU not available, skipping GPU-only tests"
        filter!(
            test ->
                !startswith(first(test), "gpu_only/") &&
                !startswith(first(test), "cuda.jl/"),
            testsuite,
        )
    end

    if !run_gpu_tests || !cuNumeric.FUSE_BROADCAST_EXPRS
        @warn "Broadcast fusion is disabled, skipping fusion tests"
        filter!(test -> !startswith(first(test), "gpu_only/broadcast_fusion"), testsuite)
    end
end

cuda_tests = filter(test -> startswith(test, "cuda.jl/"), collect(keys(testsuite)))
runtests(cuNumeric, test_args; testsuite, init_code, serial=cuda_tests)
