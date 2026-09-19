struct JACCGEMM{T}
    N::Int
    M::Int
    gpus::Int
end

struct JACCGEMMState{C,A,B}
    C::C
    A::A
    B::B
end

function model_build_gemm(config::ModelWorkerConfig)
    config.N > 0 && config.M > 0 || error("JACC GEMM dimensions must be positive")
    config.N % config.gpus == 0 || error(
        "JACC.Multi currently requires GEMM N to be divisible by the GPU count"
    )
    return JACCGEMM{config.T}(config.N, config.M, config.gpus)
end

function jacc_correctness_n(benchmark::JACCGEMM)
    return min(
        benchmark.N,
        max(benchmark.gpus, fld(8, benchmark.gpus) * benchmark.gpus),
    )
end

function jacc_gemm_state(benchmark::JACCGEMM, A, B, C)
    # MultiArray partitions matrices by columns. Repeating A horizontally gives
    # every GPU one complete A while B and C retain their column partitions.
    replicated_A = repeat(A, 1, benchmark.gpus)
    return JACCGEMMState(
        JACC.Multi.array(C), JACC.Multi.array(replicated_A), JACC.Multi.array(B)
    )
end

function model_initialize(benchmark::JACCGEMM{T}) where {T}
    available = JACC.Multi.ndev()
    available == benchmark.gpus || error(
        "JACC sees $available GPU(s), but this run was planned for $(benchmark.gpus). " *
        "Set CUDA_VISIBLE_DEVICES to exactly the selected devices.",
    )
    A = rand(T, benchmark.N, benchmark.M)
    B = rand(T, benchmark.M, benchmark.N)
    C = zeros(T, benchmark.N, benchmark.N)
    return jacc_gemm_state(benchmark, A, B, C)
end

@inline function jacc_gemm_kernel(i, j, C, A, B, inner)
    acc = @inbounds A[i, 1] * B[1, j]
    for k in 2:inner
        acc = @inbounds muladd(A[i, k], B[k, j], acc)
    end
    @inbounds C[i, j] = acc
    return nothing
end

function model_run!(benchmark::JACCGEMM, state::JACCGEMMState)
    JACC.Multi.parallel_for(
        (benchmark.N, benchmark.N), jacc_gemm_kernel,
        state.C, state.A, state.B, benchmark.M,
    )
    return state.C
end

# Multi.parallel_for synchronizes every participating device before return.
model_synchronize(::JACCGEMM) = nothing

function model_correctness_context(benchmark::JACCGEMM, config)
    return (; reference="CPU", dims=(
        jacc_correctness_n(benchmark), min(benchmark.M, 8)
    ))
end

function model_check_correctness(benchmark::JACCGEMM{T}, config) where {T}
    n, m = jacc_correctness_n(benchmark), min(benchmark.M, 8)
    check_benchmark = JACCGEMM{T}(n, m, benchmark.gpus)
    A = reshape(T.(1:(n * m)), n, m) ./ T(n*m)
    B = reshape(T.(1:(m * n)), m, n) ./ T(m*n)
    state = jacc_gemm_state(check_benchmark, A, B, zeros(T, n, n))
    model_run!(check_benchmark, state)
    actual = JACC.to_host(state.C)
    expected = A * B
    tolerance = T <: Float32 ? 1.0f-3 : 1e-10
    return isapprox(actual, expected; atol=tolerance, rtol=tolerance) ? "pass" : "fail"
end
