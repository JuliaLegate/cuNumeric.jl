Base.@kwdef struct MonteCarloIntegration{T} <: AbstractBenchmark{T}
    n_samples::Int
end

name(::MonteCarloIntegration) = "montecarlo"
dims(mci::MonteCarloIntegration) = (mci.n_samples, 1)
function data(mci::MonteCarloIntegration{T}) where {T}
    "Monte Carlo Integration with T=$(T), n_samples=$(mci.n_samples)"
end

allowed_types(::Type{MonteCarloIntegration}) = cuNumeric.SUPPORTED_FLOAT_TYPES

total_space(s::MonteCarloIntegration{T}) where {T} = s.n_samples * sizeof(T)
total_flops(s::MonteCarloIntegration) = s.n_samples

function initialize(mci::MonteCarloIntegration{T}; mod=cuNumeric) where {T}
    # Uniform samples over the integration domain [0, 10].
    x = T(10) .* mod.rand(T, mci.n_samples)
    GC.gc()
    return (x,)
end

_domain_volume(mci::MonteCarloIntegration{T}) where {T} = T(10) / mci.n_samples
run!(mci::MonteCarloIntegration, x) = _domain_volume(mci) * sum(exp.(-x .^ 2))

# n_samples comes in as N; M and variant are unused.
function build_benchmark(::Type{MonteCarloIntegration}, ::Type{T}, N, M, variant) where {T}
    MonteCarloIntegration{T}(; n_samples=N)
end

register_benchmark("montecarlo", MonteCarloIntegration)
