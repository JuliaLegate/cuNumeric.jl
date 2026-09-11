using Test

# Run with julia --project=benchmark benchmark/test/autosize.jl; no GPU needed.
include("../src/core.jl")
include("../src/benchmarks/montecarlo.jl")

@testset "Fused Monte Carlo autosizing reserves broadcast output" begin
    budget = 113_066_115_072 # Budget from the reported initialization OOM.
    for T in (Float32, Float64)
        N, M = fit_one_gpu(MonteCarloIntegration, T; budget)
        @test M == 1
        @test N % 8 == 0
        # Check actual array requirements, independently of total_space.
        @test 2 * N * sizeof(T) <= budget
        @test 2 * (N + 8) * sizeof(T) > budget
        for P in (1, 2, 4, 8)
            n, m = estimate_scaling(MonteCarloIntegration{T}(; n_samples=N), P)
            @test n == N * P
            @test m == 1
            @test 2 * n * sizeof(T) <= P * budget
        end
        @test_throws ErrorException fit_one_gpu(
            MonteCarloIntegration, T; budget=2 * 8 * sizeof(T) - 1,
        )
    end
end
