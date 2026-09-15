# Keep the ordinary array implementation in the shared benchmark file for CPU
# correctness. The cuNumeric worker replaces only its NDArray path with the
# recommended accelerated scope.
let body = quote
        integrand = _montecarlo_integrand(x)
        return _domain_volume(mci) * sum(integrand)
    end
    definition = _define_accelerated_definition(
        :(run!(mci::MonteCarloIntegration, x::NDArray)), body
    )
    @eval $definition
end
