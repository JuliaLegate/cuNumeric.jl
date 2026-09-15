struct DaggerGEMM{T,S,P}
    N::Int
    M::Int
    gpus::Int
    scope::S
    processors::P
end

struct DaggerGEMMState{C,A,B}
    C::C
    A::A
    B::B
end

function model_build_gemm(config::ModelWorkerConfig)
    config.N > 0 && config.M > 0 || error("Dagger GEMM dimensions must be positive")
    config.N >= config.gpus || error("Dagger GEMM requires N >= the GPU count")
    available = length(collect(CUDA.devices()))
    available == config.gpus || error(
        "Dagger sees $available GPU(s), but this run was planned for $(config.gpus). " *
        "Set CUDA_VISIBLE_DEVICES to exactly the selected devices.",
    )
    scope = Dagger.scope(; cuda_gpus=collect(1:config.gpus))
    processors = sort!(collect(Dagger.compatible_processors(scope)); by=string)
    length(processors) == config.gpus || error(
        "Dagger CUDA scope contains $(length(processors)) processor(s), " *
        "expected $(config.gpus)",
    )
    return DaggerGEMM{config.T,typeof(scope),typeof(processors)}(
        config.N, config.M, config.gpus, scope, processors
    )
end

function dagger_correctness_n(benchmark::DaggerGEMM)
    return min(
        benchmark.N,
        max(benchmark.gpus, fld(8, benchmark.gpus) * benchmark.gpus),
    )
end

function dagger_gemm_layout(benchmark::DaggerGEMM)
    p = benchmark.gpus
    row_block = cld(benchmark.N, p)
    col_block = cld(benchmark.N, p)
    A_assignment = reshape(copy(benchmark.processors), p, 1)
    B_assignment = reshape(copy(benchmark.processors), 1, p)
    C_assignment = [benchmark.processors[j] for _ in 1:p, j in 1:p]
    return row_block, col_block, A_assignment, B_assignment, C_assignment
end

function dagger_gemm_random_state(benchmark::DaggerGEMM{T}) where {T}
    row_block, col_block, A_assignment, B_assignment, C_assignment = dagger_gemm_layout(benchmark)
    A = rand(
        Dagger.Blocks(row_block, benchmark.M), T, benchmark.N, benchmark.M;
        assignment=A_assignment,
    )
    B = rand(
        Dagger.Blocks(benchmark.M, col_block), T, benchmark.M, benchmark.N;
        assignment=B_assignment,
    )
    C = zeros(
        Dagger.Blocks(row_block, col_block), T, benchmark.N, benchmark.N;
        assignment=C_assignment,
    )
    foreach(wait_for_darray, (C, A, B))
    return DaggerGEMMState(C, A, B)
end

function model_initialize(benchmark::DaggerGEMM)
    return Dagger.with_options(; scope=benchmark.scope) do
        return dagger_gemm_random_state(benchmark)
    end
end

function model_run!(benchmark::DaggerGEMM, state::DaggerGEMMState)
    Dagger.with_options(; scope=benchmark.scope) do
        return mul!(state.C, state.A, state.B)
    end
    wait_for_darray(state.C)
    return state.C
end

model_synchronize(::DaggerGEMM) = Dagger.gpu_synchronize(:CUDA)

function model_correctness_context(benchmark::DaggerGEMM, config)
    return (; reference="CPU", dims=(
        dagger_correctness_n(benchmark), min(benchmark.M, 8)
    ))
end

function dagger_gemm_correctness_state(benchmark::DaggerGEMM{T}, A, B) where {T}
    row_block, col_block, A_assignment, B_assignment, C_assignment = dagger_gemm_layout(benchmark)
    C_host = zeros(T, benchmark.N, benchmark.N)
    return Dagger.with_options(; scope=benchmark.scope) do
        C = Dagger.DArray(C_host, Dagger.Blocks(row_block, col_block), C_assignment)
        dA = Dagger.DArray(A, Dagger.Blocks(row_block, benchmark.M), A_assignment)
        dB = Dagger.DArray(B, Dagger.Blocks(benchmark.M, col_block), B_assignment)
        foreach(wait_for_darray, (C, dA, dB))
        return DaggerGEMMState(C, dA, dB)
    end
end

function model_check_correctness(benchmark::DaggerGEMM{T}, config) where {T}
    n, m = dagger_correctness_n(benchmark), min(benchmark.M, 8)
    check_benchmark = DaggerGEMM{
        T,typeof(benchmark.scope),typeof(benchmark.processors)
    }(
        n, m, benchmark.gpus, benchmark.scope, benchmark.processors
    )
    A = reshape(T.(1:(n * m)), n, m) ./ T(n*m)
    B = reshape(T.(1:(m * n)), m, n) ./ T(m*n)
    state = dagger_gemm_correctness_state(check_benchmark, A, B)
    model_run!(check_benchmark, state)
    model_synchronize(check_benchmark)
    actual = collect(state.C)
    expected = A * B
    tolerance = T <: Float32 ? 1.0f-3 : 1e-10
    return isapprox(actual, expected; atol=tolerance, rtol=tolerance) ? "pass" : "fail"
end
