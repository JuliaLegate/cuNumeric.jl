include(joinpath(@__DIR__, "..", "..", "nas", "mg.jl"))

Base.@kwdef struct NASMultiGrid{T} <: AbstractBenchmark{T}
    N::Int
    M::Int
    class::String = "S"
end

name(::NASMultiGrid) = "nas_mg"
dims(b::NASMultiGrid) = (b.N, b.M)
allowed_types(::Type{<:NASMultiGrid}) = Float64

function data(b::NASMultiGrid)
    p = nas_mg_parameters(b.class)
    return "NAS MG class $(uppercase(b.class)): $(p.n)^3, NITER=$(p.niter)"
end

function validate_nas_mg(b::NASMultiGrid{T}) where {T}
    T === Float64 || error("NAS MG is defined in Float64; got $T")
    p = nas_mg_parameters(b.class)
    (b.N, b.M) == (p.n, p.n) || error(
        "NAS MG class $(uppercase(b.class)) requires N=M=$(p.n); " *
        "got N=$(b.N), M=$(b.M)",
    )
    return p
end

function total_flops(b::NASMultiGrid)
    p = validate_nas_mg(b)
    return 58.0*p.niter*Float64(p.n)^3
end

function total_space(b::NASMultiGrid)
    p = validate_nas_mg(b)
    hierarchy = sum(big(n + 2)^3 for n in (2^level for level in 1:round(Int, log2(p.n))))
    return (2hierarchy + big(p.n + 2)^3)*sizeof(Float64)
end

estimate_scaling(b::NASMultiGrid, ::Integer) = dims(b)
register_benchmark("nas_mg", NASMultiGrid)
