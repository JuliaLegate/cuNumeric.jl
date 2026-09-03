Base.@kwdef struct MonteCarloIntegration{T} <: AbstractBenchmark{T}
    n_samples::Int
end

name(::MonteCarloIntegration) = "montecarlo"
dims(mci::MonteCarloIntegration) = (mci.n_samples, 1)
function data(mci::MonteCarloIntegration{T}) where {T}
    return "Monte Carlo Integration with T=$(T), n_samples=$(mci.n_samples)"
end

allowed_types(::Type{MonteCarloIntegration}) = cuNumeric.SUPPORTED_FLOAT_TYPES

total_flops(s::MonteCarloIntegration) = s.n_samples
total_space(s::MonteCarloIntegration{T}) where {T} = s.n_samples * sizeof(T)

function estimate_scaling(s::MonteCarloIntegration, P::Integer)
    P == 1 && return dims(s)
    return (s.n_samples * P, 1)
end

function fit_one_gpu(
    ::Type{MonteCarloIntegration}, ::Type{T};
    budget::Int, N_hint=nothing, M_hint=nothing,
) where {T}
    hi = max(8, Int(fld(budget, sizeof(T))))
    n = largest_feasible(8, hi, k -> total_space(MonteCarloIntegration{T}(; n_samples=k)) <= budget)
    n === nothing && error("montecarlo does not fit in $(budget) bytes")
    return (align8(n), 1)
end

function initialize(mci::MonteCarloIntegration{T}; mod=cuNumeric) where {T}
    # Uniform samples over the integration domain [0, 10].
    x = T(10) .* rand_array(mod, T, mci.n_samples)
    GC.gc()
    return (x,)
end

_domain_volume(mci::MonteCarloIntegration{T}) where {T} = T(10) / mci.n_samples
run!(mci::MonteCarloIntegration, x) = _domain_volume(mci) * sum(exp.(-x .^ 2))

# n_samples comes in as N; M is unused.
function build_benchmark(::Type{MonteCarloIntegration}, ::Type{T}, N, M) where {T}
    return MonteCarloIntegration{T}(; n_samples=N)
end

function correctness_problem(b::MonteCarloIntegration{T}) where {T}
    return MonteCarloIntegration{T}(; n_samples=min(b.n_samples, 1024))
end

register_benchmark("montecarlo", MonteCarloIntegration)
