# LIMITATION: Dagger supplies a native distributed 3-D FFT with slab/pencil
# redistributions, but not NPB's 46-bit RNG or indexed checksum reduction.
# Exact initialization and index-map construction therefore run on the host
# inside the timed sample and are copied into DArrays; each checksum is a full
# masked DArray reduction. Checksums remain as 1×1×1 DArrays until verification,
# so the runtime is not paused between FT iterations.

include(joinpath(@__DIR__, "..", "..", "..", "nas", "ft.jl"))

struct DaggerNASFT{S,P}
    class::String
    N::Int
    M::Int
    gpus::Int
    scope::S
    processors::P
end

struct DaggerNASFTState{A,H,T}
    u0::A
    u1::A
    twiddle::T
    mask::T
    host_initial::H
    host_twiddle::Array{Float64,3}
end

function model_build_nas_ft(config::ModelWorkerConfig)
    config.T === Float64 || error("NAS FT requires Float64")
    class = uppercase(string(get(config.kwargs, :class, "S")))
    p = nas_ft_parameters(class)
    (config.N, config.M) == (p.nx, p.ny) || error(
        "NAS FT class $class requires N=$(p.nx), M=$(p.ny)"
    )
    available = length(collect(CUDA.devices()))
    available == config.gpus || error(
        "Dagger sees $available GPU(s), but this run requested $(config.gpus)"
    )
    scope = Dagger.scope(; cuda_gpus=collect(1:config.gpus))
    processors = sort!(collect(Dagger.compatible_processors(scope)); by=string)
    length(processors) == config.gpus || error("Dagger CUDA processor count mismatch")
    return DaggerNASFT(class, config.N, config.M, config.gpus, scope, processors)
end

function model_initialize(b::DaggerNASFT)
    p = nas_ft_parameters(b.class)
    shape = (p.nx, p.ny, p.nz)
    blocks = Dagger.Blocks(p.nx, p.ny, cld(p.nz, b.gpus))
    assignment = reshape(copy(b.processors), 1, 1, b.gpus)
    return Dagger.with_options(; scope=b.scope) do
        u0 = Dagger.DArray(zeros(ComplexF64, shape), blocks, assignment)
        u1 = Dagger.DArray(zeros(ComplexF64, shape), blocks, assignment)
        twiddle = Dagger.DArray(zeros(Float64, shape), blocks, assignment)
        mask = Dagger.DArray(nas_ft_checksum_mask(p), blocks, assignment)
        foreach(wait_for_darray, (u0, u1, twiddle, mask))
        return DaggerNASFTState(
            u0, u1, twiddle, mask, Array{ComplexF64}(undef, shape),
            Array{Float64}(undef, shape),
        )
    end
end

function model_run!(b::DaggerNASFT, s::DaggerNASFTState)
    p = nas_ft_parameters(b.class)
    nas_ft_initial_conditions!(s.host_initial)
    nas_ft_twiddle!(s.host_twiddle)
    return Dagger.with_options(; scope=b.scope) do
        copyto!(s.u0, s.host_initial)
        copyto!(s.twiddle, s.host_twiddle)
        fft!(s.u0, (1, 2, 3); decomp=:slab)
        checksums = Any[]
        for _ in 1:p.niter
            s.u0 .*= s.twiddle
            copyto!(s.u1, s.u0)
            ifft!(s.u1, (1, 2, 3); decomp=:slab)
            push!(checksums, sum(s.u1 .* s.mask; dims=(1, 2, 3)))
        end
        # Submit every deferred checksum before the trial-level GPU fence.
        # This is one end-of-run wait, not a synchronization between iterations.
        foreach(checksum -> foreach(wait, checksum.chunks), checksums)
        return checksums
    end
end

model_synchronize(::DaggerNASFT) = Dagger.gpu_synchronize(:CUDA)

function model_check_correctness(b::DaggerNASFT, config)
    results = model_run!(b, model_initialize(b))
    model_synchronize(b)
    got = ComplexF64[only(collect(x)) for x in results]
    return nas_ft_verified(b.class, got) ? "pass" : "fail"
end

function model_correctness_context(b::DaggerNASFT, config)
    return (; reference="NPB-GPU", dims=(b.N, b.M, nas_ft_parameters(b.class).nz))
end
