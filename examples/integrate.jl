integrand = (x) -> exp(-x^2)

N = 1_000_000

x_max = 10.0f0
domain = [-x_max, x_max]
Ω = domain[2] - domain[1]

estimate = @accelerate begin
    samples = @. Ω * cuNumeric.rand(N) - x_max

    # Reductions return 0D NDArrays instead
    # of a scalar to avoid blocking runtime
    return (Ω / N) * sum(integrand.(samples))
end

println("Monte-Carlo Estimate: $(estimate)")
println("Analytical: $(sqrt(pi))")
