# Monte-Carlo Integration

For uniformly sampled points `x_i` in a domain of volume `\Omega`, a basic Monte-Carlo estimator is

```math
\hat{I}_N = \frac{\Omega}{N}\sum_{i=1}^N f(x_i).
```

This example estimates `\int_{-\infty}^{\infty} e^{-x^2}\,dx` by sampling the finite interval `[-10, 10]`. `@accelerate` frees the non-returned sample arrays after their final use. CUDA may also fuse eligible broadcasts.

```julia
# examples/integrate.jl
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
println("Monte-Carlo estimate: $(fetch(estimate))")
println("Analytical value: $(sqrt(pi))")
```

The result is a `CNScalar` backed by a 0D NDArray, which keeps the reduction asynchronous. Use `fetch(estimate)` only when a Julia scalar is required.

This can also be done with `mapreduce`.

```julia
Ω = 2 * x_max
raw_samples = cuNumeric.rand(N)
samples = @. Ω * raw_samples - x_max
estimate = (Ω / N) * mapreduce(integrand, +, samples; init = 0.0f0)
```

Each variants provides different fusion semantics and will therefore have different performance. In the `@accelerate` variant, the creation of `samples` and the application of `integrand` are fused. The application of `sum` happens after. In the `mapreduce` variant, `samples` is created first and then the `sum` and `integrand` are fused into one `mapreduce` kernel.
