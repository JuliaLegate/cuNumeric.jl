# LIMITATION: NPB-GPU uses a hand-written Stockham FFT, whereas this CUDA.jl
# baseline uses cuFFT through AbstractFFTs. Initial-condition generation,
# index-map construction, evolution, and checksum sampling remain device
# kernels, and a timed run covers the complete NPB FT execution. CUDA.jl is
# intentionally single-GPU in this harness.

struct CUDANASFTState{A,T,S,I,C}
    u0::A
    u1::A
    twiddle::T
    starts::S
    host_starts::Vector{Float64}
    indices::I
    samples::C
    checksums::Vector{C}
end

function initialize(b::NASFourierTransform{Float64}; mod=CUDA)
    p = validate_nas_ft(b)
    shape = (p.nx, p.ny, p.nz)
    u0 = CUDA.zeros(ComplexF64, shape)
    u1 = similar(u0)
    twiddle = CUDA.zeros(Float64, shape)
    starts = CUDA.zeros(Float64, p.nz)
    indices = CUDA.CuArray(Int64.(nas_ft_checksum_indices(p)))
    samples = CUDA.zeros(ComplexF64, NAS_FT_CHECKSUM_SAMPLES)
    checksums = [CUDA.zeros(ComplexF64, 1) for _ in 1:p.niter]
    return (
        CUDANASFTState(
            u0, u1, twiddle, starts, Vector{Float64}(undef, p.nz),
            indices, samples, checksums,
        ),
    )
end

@inline function cuda_nas_ft_randlc(x, a)
    r23, t23 = 2.0^-23, 2.0^23
    r46, t46 = 2.0^-46, 2.0^46
    t1 = r23*a
    a1 = trunc(t1)
    a2 = a - t23*a1
    t1 = r23*x
    x1 = trunc(t1)
    x2 = x - t23*x1
    t1 = a1*x2 + a2*x1
    t2 = trunc(r23*t1)
    z = t1 - t23*t2
    t3 = t23*z + a2*x2
    t4 = trunc(r46*t3)
    next = t3 - t46*t4
    return next, r46*next
end

function cuda_nas_ft_initial_kernel!(out, starts, plane)
    k = (CUDA.blockIdx().x - 1)*CUDA.blockDim().x + CUDA.threadIdx().x
    if k <= length(starts)
        x = @inbounds starts[k]
        offset = (k - 1)*plane
        @inbounds for i in 1:plane
            x, realpart = cuda_nas_ft_randlc(x, NAS_FT_MULTIPLIER)
            x, imagpart = cuda_nas_ft_randlc(x, NAS_FT_MULTIPLIER)
            out[offset + i] = ComplexF64(realpart, imagpart)
        end
    end
    return nothing
end

function cuda_nas_ft_twiddle_kernel!(twiddle, nx, ny, nz, ap)
    index = (CUDA.blockIdx().x - 1)*CUDA.blockDim().x + CUDA.threadIdx().x
    if index <= length(twiddle)
        zero_index = index - 1
        i = zero_index % nx
        j = (zero_index ÷ nx) % ny
        k = zero_index ÷ (nx*ny)
        ii = (i + nx÷2) % nx - nx÷2
        jj = (j + ny÷2) % ny - ny÷2
        kk = (k + nz÷2) % nz - nz÷2
        @inbounds twiddle[index] = exp(ap*(ii*ii + jj*jj + kk*kk))
    end
    return nothing
end

function cuda_nas_ft_gather_kernel!(samples, values, indices)
    j = (CUDA.blockIdx().x - 1)*CUDA.blockDim().x + CUDA.threadIdx().x
    if j <= length(samples)
        @inbounds samples[j] = values[indices[j]]
    end
    return nothing
end

function run!(b::NASFourierTransform, s::CUDANASFTState)
    p = nas_ft_parameters(b.class)
    nas_ft_plane_starts!(s.host_starts, p.nx, p.ny)
    copyto!(s.starts, s.host_starts)
    threads = 256
    CUDA.@cuda threads=threads blocks=cld(p.nz, threads) cuda_nas_ft_initial_kernel!(
        s.u0, s.starts, p.nx*p.ny
    )
    CUDA.@cuda threads=threads blocks=cld(length(s.twiddle), threads) cuda_nas_ft_twiddle_kernel!(
        s.twiddle, p.nx, p.ny, p.nz, -4.0*NAS_FT_ALPHA*pi^2
    )
    fft!(s.u0)
    for iter in 1:p.niter
        s.u0 .*= s.twiddle
        copyto!(s.u1, s.u0)
        ifft!(s.u1) # normalized inverse; NPB divides the unnormalized checksum by NTOTAL
        CUDA.@cuda threads=threads blocks=cld(length(s.samples), threads) cuda_nas_ft_gather_kernel!(
            s.samples, s.u1, s.indices
        )
        Base.mapreducedim!(identity, +, s.checksums[iter], s.samples)
    end
    return s.checksums
end

function check_benchmark_correctness(
    b::NASFourierTransform, gs::GlobalSettings; mod=CUDA
)
    state = only(initialize(b; mod))
    got = ComplexF64[only(Array(x)) for x in run!(b, state)]
    return nas_ft_verified(b.class, got) ? "pass" : "fail"
end
