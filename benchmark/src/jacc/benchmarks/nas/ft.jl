# LIMITATION: JACC has no FFT API. This single-GPU implementation therefore
# applies cuFFT through AbstractFFTs to JACC's CUDA-backed arrays. The NPB RNG,
# index map, evolution, and checksum are JACC parallel_for/parallel_reduce
# kernels and remain ordered on JACC's CUDA stream without per-iteration host
# scalar fetches. Results are useful as a documented hybrid, not pure JACC FT.
# Plane-start seeds are computed/uploaded on the host inside timing; one GPU
# thread generates each plane, as in the CUDA.jl adapter. Real and imaginary
# checksums use separate 1024-element reductions. ifft! normalizes the full array.

include(joinpath(@__DIR__, "..", "..", "..", "nas", "ft.jl"))

struct JACCNASFT
    class::String
    N::Int
    M::Int
end

struct JACCNASFTState
    u0
    u1
    twiddle
    starts
    host_starts::Vector{Float64}
    checksum_real
    checksum_imag
    real_reducer
    imag_reducer
    initial_spec
    index_spec
    evolve_spec
    save_spec
end

function model_build_nas_ft(config::ModelWorkerConfig)
    config.T === Float64 || error("NAS FT requires Float64")
    config.gpus == 1 || error("JACC NAS FT currently supports one GPU")
    class = uppercase(string(get(config.kwargs, :class, "S")))
    p = nas_ft_parameters(class)
    (config.N, config.M) == (p.nx, p.ny) || error(
        "NAS FT class $class requires N=$(p.nx), M=$(p.ny)"
    )
    return JACCNASFT(class, config.N, config.M)
end

function model_initialize(b::JACCNASFT)
    p = nas_ft_parameters(b.class)
    u0 = JACC.zeros(ComplexF64, p.nx, p.ny, p.nz)
    u1 = JACC.zeros(ComplexF64, p.nx, p.ny, p.nz)
    twiddle = JACC.zeros(Float64, p.nx, p.ny, p.nz)
    starts = JACC.zeros(Float64, p.nz)
    return JACCNASFTState(
        u0, u1, twiddle, starts, Vector{Float64}(undef, p.nz),
        JACC.zeros(Float64, p.niter), JACC.zeros(Float64, p.niter),
        JACC.reducer(; range=NAS_FT_CHECKSUM_SAMPLES, type=Float64, sync=false),
        JACC.reducer(; range=NAS_FT_CHECKSUM_SAMPLES, type=Float64, sync=false),
        JACC.launch_spec(; sync=false, shmem_size=0), JACC.launch_spec(; sync=false, shmem_size=0),
        JACC.launch_spec(; sync=false, shmem_size=0), JACC.launch_spec(; sync=false, shmem_size=0),
    )
end

@inline function jacc_nas_ft_randlc(x, a)
    r23, t23 = 2.0^-23, 2.0^23
    r46, t46 = 2.0^-46, 2.0^46
    a1 = trunc(r23*a)
    a2 = a - t23*a1
    x1 = trunc(r23*x)
    x2 = x - t23*x1
    t1 = a1*x2 + a2*x1
    z = t1 - t23*trunc(r23*t1)
    t3 = t23*z + a2*x2
    next = t3 - t46*trunc(r46*t3)
    return next, r46*next
end

function jacc_nas_ft_initial(k, out, starts, plane)
    x = @inbounds starts[k]
    offset = (k - 1)*plane
    @inbounds for i in 1:plane
        x, realpart = jacc_nas_ft_randlc(x, NAS_FT_MULTIPLIER)
        x, imagpart = jacc_nas_ft_randlc(x, NAS_FT_MULTIPLIER)
        out[offset + i] = ComplexF64(realpart, imagpart)
    end
end

function jacc_nas_ft_indexmap(index, twiddle, nx, ny, nz, ap)
    zero_index = index - 1
    i = zero_index % nx
    j = (zero_index ÷ nx) % ny
    k = zero_index ÷ (nx*ny)
    ii = (i + nx÷2) % nx - nx÷2
    jj = (j + ny÷2) % ny - ny÷2
    kk = (k + nz÷2) % nz - nz÷2
    @inbounds twiddle[index] = exp(ap*(ii*ii + jj*jj + kk*kk))
end

function jacc_nas_ft_evolve(index, u0, u1, twiddle)
    @inbounds u0[index] *= twiddle[index]
    @inbounds u1[index] = u0[index]
end

@inline function jacc_nas_ft_sample_index(j, nx, ny, nz)
    q, r, s = j % nx, (3j) % ny, (5j) % nz
    return q + r*nx + s*nx*ny + 1
end

function jacc_nas_ft_checksum_real(j, values, nx, ny, nz)
    return @inbounds real(values[jacc_nas_ft_sample_index(j, nx, ny, nz)])
end

function jacc_nas_ft_checksum_imag(j, values, nx, ny, nz)
    return @inbounds imag(values[jacc_nas_ft_sample_index(j, nx, ny, nz)])
end

function jacc_nas_ft_save_checksum(_, re, im, iter, re_value, im_value)
    @inbounds re[iter] = re_value[1]
    @inbounds im[iter] = im_value[1]
end

function model_run!(b::JACCNASFT, s::JACCNASFTState)
    p = nas_ft_parameters(b.class)
    n = p.nx*p.ny*p.nz
    nas_ft_plane_starts!(s.host_starts, p.nx, p.ny)
    copyto!(s.starts, s.host_starts)
    JACC.parallel_for(
        s.initial_spec, p.nz, jacc_nas_ft_initial, s.u0, s.starts, p.nx*p.ny
    )
    JACC.parallel_for(
        s.index_spec, n, jacc_nas_ft_indexmap, s.twiddle,
        p.nx, p.ny, p.nz, -4.0*NAS_FT_ALPHA*pi^2,
    )
    fft!(s.u0)
    for iter in 1:p.niter
        JACC.parallel_for(s.evolve_spec, n, jacc_nas_ft_evolve, s.u0, s.u1, s.twiddle)
        ifft!(s.u1)
        s.real_reducer(jacc_nas_ft_checksum_real, s.u1, p.nx, p.ny, p.nz)
        s.imag_reducer(jacc_nas_ft_checksum_imag, s.u1, p.nx, p.ny, p.nz)
        JACC.parallel_for(
            s.save_spec, 1, jacc_nas_ft_save_checksum,
            s.checksum_real, s.checksum_imag, iter,
            s.real_reducer.workspace.ret, s.imag_reducer.workspace.ret,
        )
    end
    return s.checksum_real, s.checksum_imag
end

model_synchronize(::JACCNASFT) = JACC.synchronize()

function model_check_correctness(b::JACCNASFT, config)
    state = model_initialize(b)
    model_run!(b, state)
    model_synchronize(b)
    re, im = JACC.to_host(state.checksum_real), JACC.to_host(state.checksum_imag)
    return nas_ft_verified(b.class, complex.(re, im)) ? "pass" : "fail"
end

function model_correctness_context(b::JACCNASFT, config)
    return (; reference="NPB-GPU", dims=(b.N, b.M, nas_ft_parameters(b.class).nz))
end
