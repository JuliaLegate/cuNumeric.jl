# Minimal runner for test/tests/diagonal.jl only.
using Test
using LinearAlgebra
using Random
using cuNumeric

include(joinpath(@__DIR__, "tests", "util.jl"))

@testset verbose = true "Diagonal / UniformScaling" begin
    include(joinpath(@__DIR__, "tests", "diagonal.jl"))
end
