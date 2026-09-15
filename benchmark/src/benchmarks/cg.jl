Base.@kwdef struct ConjugateGradientBenchmark{T} <: AbstractBenchmark{T}
    N::Int
    M::Int = 1
    check_every::Int = 10
    max_iter::Int = 1000
end
name(::ConjugateGradientBenchmark) = "cg"
dims(b::ConjugateGradientBenchmark) = (b.N,1)
data(b::ConjugateGradientBenchmark) = "CG: N=$(b.N), check_every=$(b.check_every), max_iter=$(b.max_iter)"
allowed_types(::Type{<:ConjugateGradientBenchmark}) = Union{Float32,Float64}
# Executed iterations depend on convergence; compare elapsed time, not nominal FLOPs.
total_flops(::ConjugateGradientBenchmark) = 0
estimate_scaling(b::ConjugateGradientBenchmark,p::Integer) = (scale_axis(b.N,p,1),1)
total_space(b::ConjugateGradientBenchmark{T}) where {T} = 7big(b.N)*sizeof(T)
correctness_uses_cpu(::ConjugateGradientBenchmark) = true
register_benchmark("cg",ConjugateGradientBenchmark)
