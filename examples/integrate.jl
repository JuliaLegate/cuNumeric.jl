using cuNumeric

integrand(x) = @. exp(-x^2)

@accelerate function monte_carlo(N, x_max)
    Ω = 2 * x_max
    raw_samples = cuNumeric.rand(N)
    samples = @. Ω * raw_samples - x_max
    # Reductions return 0D NDArrays to avoid blocking the runtime.
    return (Ω / N) * sum(integrand(samples))
end

if abspath(PROGRAM_FILE) == @__FILE__
    estimate = monte_carlo(1_000_000, 10.0f0)
    println("Monte-Carlo estimate: $(estimate)")
    println("Analytical value: $(sqrt(pi))")
end
