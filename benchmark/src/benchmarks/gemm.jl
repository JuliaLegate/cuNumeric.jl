# Interface: `name`, `dims`, `total_flops`, `total_space`, `estimate_scaling`,
# `fit_one_gpu`, `initialize`, `run!`. Peak-byte and P-scaling formulas live
# here so the orchestrator can dispatch without a name switch.

Base.@kwdef struct GEMM{T} <: AbstractBenchmark{T}
    N::Int
    M::Int
end

name(::GEMM) = "gemm"
dims(g::GEMM) = (g.N, g.M)
data(g::GEMM{T}) where {T} = "GEMM with T=$(T), N=$(g.N), M=$(g.M)"

function allowed_types(::Type{GEMM})
    return Union{cuNumeric.SUPPORTED_FLOAT_TYPES,cuNumeric.SUPPORTED_INT_TYPES}
end

total_flops(s::GEMM) = s.N * s.N * ((2*s.M) - 1)

# Live arrays for `mul!(C, A, B)`: A (N×M), B (M×N), C (N×N).
total_space(s::GEMM{T}) where {T} = (2 * s.N * s.M + s.N * s.N) * sizeof(T)

function estimate_scaling(s::GEMM, P::Integer)
    P == 1 && return (s.N, s.M)
    n = scale_axis(s.N, P, 1//3)
    return (n, n)
end

function fit_one_gpu(
    ::Type{GEMM}, ::Type{T};
    budget::Int, N_hint=nothing, M_hint=nothing,
) where {T}
    hi = max(8, Int(floor(sqrt(Float64(budget) / sizeof(T)))))
    n = largest_feasible(8, hi, k -> total_space(GEMM{T}(; N=k, M=k)) <= budget)
    n === nothing && error("gemm does not fit in $(budget) bytes")
    n = align8(n)
    return (n, n)
end

function initialize(s::GEMM{T}; mod=cuNumeric) where {T}
    A = rand_array(mod, T, s.N, s.M)
    B = rand_array(mod, T, s.M, s.N)
    C = zeros_array(mod, T, s.N, s.N)
    GC.gc()
    return C, A, B
end

run!(::GEMM, C, A, B) = mul!(C, A, B)

correctness_problem(b::GEMM{T}) where {T} = GEMM{T}(; N=min(b.N, 8), M=min(b.M, 8))

register_benchmark("gemm", GEMM)
