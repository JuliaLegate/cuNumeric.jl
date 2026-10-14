using cuNumeric

integrand(x) = exp(-x^2)

@accelerate function monte_carlo(N, x_max)
    Ω = 2 * x_max
    raw_samples = cuNumeric.rand(N)
    samples = @. Ω * raw_samples - x_max
    return (Ω / N) * sum(integrand.(samples))
end

N = 1_000_000
x_max = 10.0f0
estimate = monte_carlo(N, x_max)
println("Monte-Carlo estimate: $(estimate)")
println("Analytical value: $(sqrt(pi))")

# Alternative: fuse the integrand and reduction with mapreduce.
Ω = 2 * x_max
raw_samples = cuNumeric.rand(N)
samples = @. Ω * raw_samples - x_max
estimate = (Ω / N) * mapreduce(integrand, +, samples; init = 0.0f0)
println("Monte-Carlo estimate (mapreduce): $(estimate)")
