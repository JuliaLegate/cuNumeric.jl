using Printf
using Statistics

# Adding a benchmark is: drop a file in benchmarks/ and include it below.
include("core.jl")
include("benchmarks/gemm.jl")
include("benchmarks/grayscott.jl")
include("benchmarks/montecarlo.jl")
