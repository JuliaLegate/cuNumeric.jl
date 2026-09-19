include(joinpath(@__DIR__, "..", "..", "nas", "ft.jl"))

Base.@kwdef struct NASFourierTransform{T} <: AbstractBenchmark{T}
    N::Int
    M::Int
    class::String = "S"
end

name(::NASFourierTransform) = "nas_ft"
dims(b::NASFourierTransform) = (b.N, b.M)
allowed_types(::Type{<:NASFourierTransform}) = Float64

function data(b::NASFourierTransform)
    p = nas_ft_parameters(b.class)
    return "NAS FT class $(uppercase(b.class)): $(p.nx)×$(p.ny)×$(p.nz), NITER=$(p.niter)"
end

function validate_nas_ft(b::NASFourierTransform{T}) where {T}
    T === Float64 || error("NAS FT is defined in Float64; got $T")
    p = nas_ft_parameters(b.class)
    (b.N, b.M) == (p.nx, p.ny) || error(
        "NAS FT class $(uppercase(b.class)) requires N=$(p.nx), M=$(p.ny); " *
        "got N=$(b.N), M=$(b.M)",
    )
    return p
end

function total_flops(b::NASFourierTransform)
    p = validate_nas_ft(b)
    ntotal = Float64(p.nx)*p.ny*p.nz
    l = log(ntotal)
    return ntotal * (14.8157 + 7.19641*l + (5.23518 + 7.21113*l)*p.niter)
end

function total_space(b::NASFourierTransform)
    p = validate_nas_ft(b)
    n = big(p.nx)*p.ny*p.nz
    return 3n*sizeof(ComplexF64) + 2n*sizeof(Float64)
end

estimate_scaling(b::NASFourierTransform, ::Integer) = dims(b)
register_benchmark("nas_ft", NASFourierTransform)
