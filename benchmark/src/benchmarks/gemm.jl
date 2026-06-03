Base.@kwdef struct GEMM{T} <: AbstractBenchmark{T}
    N::Int
    M::Int
end

name(::GEMM) = "gemm"
dims(g::GEMM) = (g.N, g.M)
data(g::GEMM{T}) where {T} = "GEMM with T=$(T), N=$(g.N), M=$(g.M)"

function allowed_types(::Type{GEMM})
    Union{cuNumeric.SUPPORTED_FLOAT_TYPES,cuNumeric.SUPPORTED_INT_TYPES}
end

total_flops(s::GEMM) = s.N * s.N * ((2*s.M) - 1)
total_space(s::GEMM{T}) where {T} = 2 * ((s.N*s.M) * sizeof(T)) + ((s.N*s.N) * sizeof(T))

function initialize(s::GEMM{T}) where {T}
    A = cuNumeric.rand(T, s.N, s.M)
    B = cuNumeric.rand(T, s.M, s.N)
    C = cuNumeric.zeros(T, s.N, s.N)
    GC.gc()
    return C, A, B
end

run!(::GEMM, C, A, B) = mul!(C, A, B)

register_benchmark("gemm", GEMM)
