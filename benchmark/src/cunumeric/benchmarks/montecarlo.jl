function run!(mci::MonteCarloIntegration{T}, x::NDArray) where {T}
    total = mapreduce(_montecarlo_scalar_integrand, +, x; init=zero(T))
    return _domain_volume(mci) * total
end

function benchmark_backend_label(
    ::MonteCarloIntegration, backend::String, default::String
)
    return backend == "cunumeric" ? "cuNumeric (mapreduce)" : default
end
