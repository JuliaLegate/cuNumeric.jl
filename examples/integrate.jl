# found in examples/integrate.jl
using cuNumeric

integrand = (x) -> exp(-square(x))

N = 1_000_000

x_max = 5.0
domain = [-x_max, x_max]
Ω = domain[2] - domain[1]

samples = Ω*cuNumeric.rand(NDArray, N) - x_max
estimate = (Ω/N) * sum(integrand(samples))

println("Monte-Carlo Estimate: $(estimate[1])")
println("Analytical: $(sqrt(pi))")
