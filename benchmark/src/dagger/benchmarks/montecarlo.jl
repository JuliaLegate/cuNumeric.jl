struct DaggerMonteCarlo{T,S}
    n_samples::Int
    gpus::Int
    scope::S
end

function model_build_benchmark(config::ModelWorkerConfig)
    config.name == "montecarlo" || error(
        "Dagger benchmark '$(config.name)' is not implemented; known: montecarlo"
    )
    available = length(collect(CUDA.devices()))
    available == config.gpus || error(
        "Dagger sees $available GPU(s), but this run was planned for $(config.gpus). " *
        "Set CUDA_VISIBLE_DEVICES to exactly the selected devices.",
    )
    scope = Dagger.scope(; cuda_gpus=collect(1:config.gpus))
    return DaggerMonteCarlo{config.T,typeof(scope)}(config.N, config.gpus, scope)
end

function wait_for_darray(array)
    foreach(wait, array.chunks)
    return array
end

function model_initialize(benchmark::DaggerMonteCarlo{T}) where {T}
    block = cld(benchmark.n_samples, benchmark.gpus)
    return Dagger.with_options(; scope=benchmark.scope) do
        samples = T(10) .* rand(Dagger.Blocks(block), T, benchmark.n_samples)
        return wait_for_darray(samples)
    end
end

function model_run!(benchmark::DaggerMonteCarlo{T}, samples) where {T}
    total = Dagger.with_options(; scope=benchmark.scope) do
        sum(samples) do x
            return exp(-(x*x))
        end
    end
    return (T(10) / benchmark.n_samples) * total
end

model_synchronize(::DaggerMonteCarlo) = Dagger.gpu_synchronize(:CUDA)
function model_correctness_context(benchmark::DaggerMonteCarlo, config)
    return (; reference="CPU", dims=(min(benchmark.n_samples, 1024), 1))
end

function model_check_correctness(benchmark::DaggerMonteCarlo{T}, config) where {T}
    n = min(benchmark.n_samples, 1024)
    host_samples = montecarlo_correctness_samples(T, n)
    block = cld(n, benchmark.gpus)
    samples = Dagger.with_options(; scope=benchmark.scope) do
        return wait_for_darray(Dagger.DArray(host_samples, Dagger.Blocks(block)))
    end
    check_benchmark = DaggerMonteCarlo{T,typeof(benchmark.scope)}(
        n, benchmark.gpus, benchmark.scope
    )
    actual = model_run!(check_benchmark, samples)
    model_synchronize(check_benchmark)
    expected = montecarlo_correctness_reference(host_samples)
    return montecarlo_correctness_status(actual, expected, T)
end
