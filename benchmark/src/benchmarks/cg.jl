abstract type AbstractConjugateGradient{T} <: AbstractBenchmark{T} end

Base.@kwdef struct ConjugateGradientBenchmark{T} <: AbstractConjugateGradient{T}
    N::Int
    M::Int = 1
    check_every::Int = 10
    max_iter::Int = 1000
end
Base.@kwdef struct ConjugateGradientAccelerated{T} <: AbstractConjugateGradient{T}
    N::Int
    M::Int = 1
    check_every::Int = 10
    max_iter::Int = 1000
end
name(::ConjugateGradientAccelerated) = "cg_accelerated"
name(::ConjugateGradientBenchmark) = "cg"
dims(b::AbstractConjugateGradient) = (b.N,1)
data(b::AbstractConjugateGradient) = "CG: N=$(b.N), check_every=$(b.check_every), max_iter=$(b.max_iter)"
allowed_types(::Type{<:AbstractConjugateGradient}) = Union{Float32,Float64}
# Executed iterations depend on convergence; compare elapsed time, not nominal FLOPs.
total_flops(::AbstractConjugateGradient) = 0
estimate_scaling(b::AbstractConjugateGradient,p::Integer) = (scale_axis(b.N,p,1),1)
total_space(b::AbstractConjugateGradient{T}) where {T} = 7big(b.N)*sizeof(T)
correctness_uses_cpu(::AbstractConjugateGradient) = true
register_benchmark("cg",ConjugateGradientBenchmark)
register_benchmark("cg_accelerated",ConjugateGradientAccelerated)
