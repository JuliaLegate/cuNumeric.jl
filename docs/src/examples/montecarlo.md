# Monte-Carlo Integration

Most integrals can be estimated with a basic Monte-Carlo estimator:

```math
\hat{I}_N = \frac{\Omega}{N}\sum_{i=1}^Nf(x_i)
```
where `N` is the number of samples, ``\Omega`` is the volume of the domain and ``x_i`` are sampled indpendently and uniformly at random from the domain. This estimator is guranteed to converge (subject to some minor constraints) at a rate independent of the dimension and is embaressingly parallel to compute!

In the example below, we estimate the integral:
```math
I = \int_{-\infty}^{\infty}e^{-x^2}.
```

Since we cannot uniformly sample form negative to positive infinity, we truncate the domain between -5 and 5. This is ok since the integrand exponentially decays and we won't be off by much in the end.
```julia
# found in examples/integrate.jl
using cuNumeric

# Note that we do not yet support broadcasting
# custom functions over NDArray, so the broadcasting MUST
# be done inside the function
integrand = (x) -> @. exp(-x^2)

N = 1_000_000

x_max = 10.0f0
domain = [-x_max, x_max]
Ω = domain[2] - domain[1]

samples = Ω * cuNumeric.rand(N)
samples = @. samples - x_max

# Reductions return 0D NDArrays instead
# of a scalar to avoid blocking runtime
estimate = (Ω / N) * sum(integrand(samples))

println("Monte-Carlo Estimate: $(estimate)")
println("Analytical: $(sqrt(pi))")
```
