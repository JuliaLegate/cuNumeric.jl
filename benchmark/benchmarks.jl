abstract type AbstractBenchmark{T} end

#########################################

Base.@kwdef struct GEMM{T} <: AbstractBenchmark{T}
    N::Int
    M::Int
end

data(g::GEMM{T}) where {T} = "GEMM with T=$(T), N=$(g.N), M=$(g.M)"

function allowed_types(::MonteCarloIntegration)
    Union{cuNumeric.SUPPORTED_FLOAT_TYPES,cuNumeric.SUPPORTED_INT_TYPES}
end

total_flops(s::GEMM) = s.N * s.N * ((2*s.M) - 1)
total_space(s::GEMM{T}) where {T} = 2 * ((s.N*s.M) * sizeof(T)) + ((s.N*s.N) * sizeof(T))

function initialize_cpu(s::GEMM{T}) where {T}
    A = rand(T, s.N, s.M)
    B = rand(T, s.M, s.N)
    C = zeros(T, s.N, s.N)
    return A, B, C
end

run!(::GEMM, C, A, B) = mul!(C, A, B)

#########################################

Base.@kwdef struct MonteCarloIntegration{T} <: AbstractBenchmark{T}
    n_samples::Int
end

function data(mci::MonteCarloIntegration{T}) where {T}
    "Monte Carlo Integration with T=$(T), n_samples=$(mci.n_samples)"
end

allowed_types(::MonteCarloIntegration) = cuNumeric.SUPPORTED_FLOAT_TYPES

total_space(s::MonteCarloIntegration{T}) where {T} = s.n_samples * sizeof(T)
total_flops(s::MonteCarloIntegration) = missing # cannot estimate FLOPS for squaring or exp easily

function initialize_cpu(s::MonteCarloIntegration{T}) where {T}
    return T(10) .* rand(T, s.n_samples) .+ T(-5) # random samples in [-5, 5]
end

_domain_volume(mci::MonteCarloIntegration{T}) where {T} = T(10) / mci.n_samples
run!(mci::MonteCarloIntegration, x) = _domain_volume(mci) * sum(exp.(-x .^ 2))

#################
