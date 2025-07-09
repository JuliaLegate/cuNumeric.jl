# using Pkg
# Pkg.activate("../")

using cuNumeric
using LinearAlgebra

function initialize_cunumeric(N)
    A = cuNumeric.as_type(cuNumeric.rand(NDArray, N, N), LegateType(Float32))
    B = cuNumeric.as_type(cuNumeric.rand(NDArray, N, N), LegateType(Float32))
    C = cuNumeric.zeros(Float32, N, N)
    return A, B, C
end

function total_flops(N)
    return N * N * ((2*N) - 1)
end

function total_space(N)
    return 3 * (N^2) * sizeof(Float32)
end

function gemm_cunumeric(N, n_samples, n_warmup)
    A, B, C = initialize_cunumeric(N)

    start_time = nothing
    for idx in range(1, n_samples + n_warmup)
        if idx == n_warmup + 1
            start_time = get_time_microseconds()
        end

        mul!(C, A, B)
    end
    total_time_μs = get_time_microseconds() - start_time
    mean_time_ms = total_time_μs / (n_samples * 1e3)
    gflops = total_flops(N) / (mean_time_ms * 1e6) # GFLOP is 1e9

    return mean_time_ms, gflops
end

# not very generic but for now whatever
N = parse(Int, ARGS[1])
n_samples = parse(Int, ARGS[2])
n_warmup = parse(Int, ARGS[3])

@info "Running MATMUL benchmark on $(N)x$N matricies for $n_samples iterations, $(n_warmup) warmups"

mean_time_ms, gflops = gemm_cunumeric(N, n_samples, n_warmup)

println("cuNumeric Mean Run Time: $(mean_time_ms) ms")
println("cuNumeric FLOPS: $(gflops) GFLOPS")

# ./run_benchmark.sh matmul.jl --cpus 10 --gpus 1 1000 25 5
