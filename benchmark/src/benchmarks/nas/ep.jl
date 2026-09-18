include(joinpath(@__DIR__, "..", "..", "nas", "ep.jl"))

Base.@kwdef struct NASEmbarrassinglyParallel{T} <: AbstractBenchmark{T}
    N::Int
    M::Int
    class::String = "S"
end

name(::NASEmbarrassinglyParallel) = "nas_ep"
dims(b::NASEmbarrassinglyParallel) = (b.N, b.M)
allowed_types(::Type{<:NASEmbarrassinglyParallel}) = Float64
throughput_label(::NASEmbarrassinglyParallel) = "G random numbers/s"

function data(b::NASEmbarrassinglyParallel)
    p = nas_ep_parameters(b.class)
    return "NAS EP class $(uppercase(b.class)): 2^$(p.m + 1) random numbers"
end

function validate_nas_ep(b::NASEmbarrassinglyParallel{T}) where {T}
    T === Float64 || error("NAS EP is defined in Float64; got $T")
    b.M == 1 || error("NAS EP requires M=1")
    p = nas_ep_parameters(b.class)
    expected = nas_ep_random_numbers(p)
    b.N == expected || error(
        "NAS EP class $(uppercase(b.class)) requires N=$expected, got $(b.N)"
    )
    return p
end

# NPB reports millions of random numbers generated per second for EP. The
# harness stores that nominal operation rate in its common throughput column.
total_flops(b::NASEmbarrassinglyParallel) = Float64(nas_ep_random_numbers(validate_nas_ep(b)))

function total_space(b::NASEmbarrassinglyParallel)
    p = validate_nas_ep(b)
    streams = big(nas_ep_batches(p))
    return streams * (13 + p.m - NAS_EP_MK) * sizeof(Float64)
end

estimate_scaling(b::NASEmbarrassinglyParallel, ::Integer) = dims(b)
register_benchmark("nas_ep", NASEmbarrassinglyParallel)
