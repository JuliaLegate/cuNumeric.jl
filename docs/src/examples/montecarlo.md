# Monte-Carlo Integration

For uniformly sampled points `x_i` in a domain of volume `\Omega`, a basic Monte-Carlo estimator is

```math
\hat{I}_N = \frac{\Omega}{N}\sum_{i=1}^N f(x_i).
```

This example estimates `\int_{-\infty}^{\infty} e^{-x^2}\,dx` by sampling the finite interval `[-10, 10]`. `@accelerate` frees the non-returned sample arrays after their final use; CUDA may also fuse eligible broadcasts.

```julia
# examples/integrate.jl
using cuNumeric

integrand(x) = @. exp(-x^2)

@accelerate function monte_carlo(N, x_max)
    Ω = 2 * x_max
    raw_samples = cuNumeric.rand(N)
    samples = @. Ω * raw_samples - x_max
    return (Ω / N) * sum(integrand(samples))
end

estimate = monte_carlo(1_000_000, 10.0f0)
println("Monte-Carlo estimate: $(estimate)")
println("Analytical value: $(sqrt(pi))")
```

The result is a 0-dimensional `NDArray`, which keeps the reduction asynchronous. Use `unwrap(estimate)` only when a Julia scalar is required.
