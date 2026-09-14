struct JACCMonteCarlo{T}
    n_samples::Int
    gpus::Int
end

function model_build_benchmark(config::ModelWorkerConfig)
    config.name == "montecarlo" || error(
        "JACC benchmark '$(config.name)' is not implemented; known: montecarlo"
    )
    config.N % config.gpus == 0 || error(
        "JACC.Multi currently requires montecarlo N to be divisible by the GPU count"
    )
    return JACCMonteCarlo{config.T}(config.N, config.gpus)
end

function model_initialize(benchmark::JACCMonteCarlo{T}) where {T}
    available = JACC.Multi.ndev()
    available == benchmark.gpus || error(
        "JACC sees $available GPU(s), but this run was planned for $(benchmark.gpus). " *
        "Set CUDA_VISIBLE_DEVICES to exactly the selected devices.",
    )
    host_samples = T(10) .* rand(T, benchmark.n_samples)
    return JACC.Multi.array(host_samples)
end

@inline function jacc_montecarlo_integrand(i, samples)
    x = @inbounds samples[i]
    return exp(-(x*x))
end

function model_run!(benchmark::JACCMonteCarlo{T}, samples) where {T}
    integral = JACC.Multi.parallel_reduce(
        benchmark.n_samples, jacc_montecarlo_integrand, samples
    )
    return (T(10) / benchmark.n_samples) * integral
end

# Multi.parallel_reduce synchronizes every participating device before return.
model_synchronize(::JACCMonteCarlo) = nothing

function model_check_correctness(benchmark::JACCMonteCarlo{T}, config) where {T}
    n = min(benchmark.n_samples, 1024)
    samples = montecarlo_correctness_samples(T, n)
    check_benchmark = JACCMonteCarlo{T}(n, benchmark.gpus)
    actual = model_run!(check_benchmark, JACC.Multi.array(samples))
    expected = montecarlo_correctness_reference(samples)
    return montecarlo_correctness_status(actual, expected, T)
end
