function run!(mci::MonteCarloIntegration{T}, x::CUDA.CuArray) where {T}
    total = mapreduce(_montecarlo_scalar_integrand, +, x; init=zero(T))
    return _domain_volume(mci) * total
end

function benchmark_backend_label(
    ::MonteCarloIntegration, backend::String, default::String
)
    return backend == "cudajl" ? "CUDA.jl (mapreduce)" : default
end
