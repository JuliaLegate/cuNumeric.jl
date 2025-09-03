using cuNumeric
using LinearAlgebra
using Printf

function initialize_cunumeric(N, M)
    A = cuNumeric.as_type(cuNumeric.rand(NDArray, N, M), Float32)
    B = cuNumeric.as_type(cuNumeric.rand(NDArray, M, N), Float32)
    C = cuNumeric.zeros(Float32, N, N)
    GC.gc() # remove the intermediate FP64 arrays
    return A, B, C
end

function total_flops(N, M)
    return N * N * ((2*M) - 1)
end

function total_space(N, M)
    return 2 * (N*M) * sizeof(Float32) + (N*N) * sizeof(Float32)
end

function gemm_cunumeric(N, M, n_samples, n_warmup)
    A, B, C = initialize_cunumeric(N, M)

    start_time = nothing
    for idx in range(1, n_samples + n_warmup)
        if idx == n_warmup + 1
            start_time = get_time_microseconds()
        end

        mul!(C, A, B)
    end
    total_time_μs = get_time_microseconds() - start_time
    mean_time_ms = total_time_μs / (n_samples * 1e3)
    gflops = total_flops(N, M) / (mean_time_ms * 1e6) # GFLOP is 1e9

    return mean_time_ms, gflops
end

gpus = parse(Int, ARGS[1])
N = parse(Int, ARGS[2])
M = parse(Int, ARGS[3])
n_samples = parse(Int, ARGS[4])
n_warmup = parse(Int, ARGS[5])

println(
    "[cuNumeric]  MATMUL benchmark on $(N)x$(M) matricies for $(n_samples) iterations, $(n_warmup) warmups",
)

mean_time_ms, gflops = gemm_cunumeric(N, M, n_samples, n_warmup)

println("[cuNumeric]  Mean Run Time: $(mean_time_ms) ms")
println("[cuNumeric]  FLOPS: $(gflops) GFLOPS")

open("./gemm.csv", "a") do io
    @printf(io, "%s,%d,%d,%d,%.6f,%.6f\n", "cunumeric", gpus, N, M, mean_time_ms, gflops)
end
