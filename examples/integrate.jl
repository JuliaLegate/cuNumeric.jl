using cuNumeric

# Note that we do not yet support broadcasting
# custom functions, so the braodcasting MUST
# be done inside the function
integrand = (x) -> exp.(-x.^2)

N = 1_000_000

x_max = 10.0f0
domain = [-x_max, x_max]
Ω = domain[2] - domain[1]

samples = Ω*cuNumeric.rand(N) .- x_max 

# Reductions return 0D NDArrays instead 
# of a scalar to avoid blocking runtime
estimate = (Ω/N) * sum(integrand(samples))

println("Monte-Carlo Estimate: $(estimate)")
println("Analytical: $(sqrt(pi))")