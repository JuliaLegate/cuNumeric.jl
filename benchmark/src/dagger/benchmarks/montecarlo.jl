struct DaggerMonteCarlo{T,S,P}
    n_samples::Int
    gpus::Int
    scope::S
    processors::P
end

struct DaggerMonteCarloState{A,C,S}
    samples::A
    chunks::C
    scopes::S
end

function model_build_montecarlo(config::ModelWorkerConfig)
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
    return DaggerMonteCarlo{config.T,typeof(scope),typeof(processors)}(
        config.N, config.gpus, scope, processors
    )
end

function dagger_montecarlo_state(samples, expected_gpus)
    chunks = map(samples.chunks) do task
        return fetch(task; raw=true)
    end
    length(chunks) == expected_gpus || error(
        "Dagger created $(length(chunks)) Monte Carlo chunk(s), expected $expected_gpus"
    )
    all(chunk -> Dagger.chunktype(chunk) <: CUDA.CuArray, chunks) || error(
        "Dagger Monte Carlo chunks must be resident CUDA arrays"
    )
    processors = Dagger.processor.(chunks)
    length(unique(processors)) == expected_gpus || error(
        "Dagger did not place exactly one Monte Carlo chunk on each requested GPU"
    )
    scopes = Dagger.ExactScope.(processors)
    return DaggerMonteCarloState(samples, chunks, scopes)
end

@inline function dagger_montecarlo_integrand(x)
    return exp(-(x*x))
end

function dagger_montecarlo_chunk_sum(samples)
    return mapreduce(
        dagger_montecarlo_integrand, +, samples; init=zero(eltype(samples))
    )
end

function model_initialize(benchmark::DaggerMonteCarlo{T}) where {T}
    block = cld(benchmark.n_samples, benchmark.gpus)
    return Dagger.with_options(; scope=benchmark.scope) do
        samples =
            T(10) .* rand(
                Dagger.Blocks(block), T, benchmark.n_samples;
                assignment=benchmark.processors,
            )
        wait_for_darray(samples)
        return dagger_montecarlo_state(samples, benchmark.gpus)
    end
end

function model_run!(benchmark::DaggerMonteCarlo{T}, state::DaggerMonteCarloState) where {T}
    partials = map(zip(state.chunks, state.scopes)) do (chunk, scope)
        Dagger.@spawn scope=scope dagger_montecarlo_chunk_sum(chunk)
    end
    total = mapreduce(fetch, +, partials; init=zero(T))
    return (T(10) / benchmark.n_samples) * total
end

function dagger_montecarlo_correctness_state(benchmark::DaggerMonteCarlo, host_samples)
    block = cld(length(host_samples), benchmark.gpus)
    return Dagger.with_options(; scope=benchmark.scope) do
        samples = Dagger.DArray(
            host_samples, Dagger.Blocks(block), benchmark.processors
        )
        wait_for_darray(samples)
        return dagger_montecarlo_state(samples, benchmark.gpus)
    end
end

model_synchronize(::DaggerMonteCarlo) = Dagger.gpu_synchronize(:CUDA)
function model_correctness_context(benchmark::DaggerMonteCarlo, config)
    return (; reference="CPU", dims=(min(benchmark.n_samples, 1024), 1))
end

function model_check_correctness(benchmark::DaggerMonteCarlo{T}, config) where {T}
    n = min(benchmark.n_samples, 1024)
    host_samples = montecarlo_correctness_samples(T, n)
    state = dagger_montecarlo_correctness_state(benchmark, host_samples)
    check_benchmark = DaggerMonteCarlo{
        T,typeof(benchmark.scope),typeof(benchmark.processors)
    }(
        n, benchmark.gpus, benchmark.scope, benchmark.processors
    )
    actual = model_run!(check_benchmark, state)
    model_synchronize(check_benchmark)
    expected = montecarlo_correctness_reference(host_samples)
    return montecarlo_correctness_status(actual, expected, T)
end
