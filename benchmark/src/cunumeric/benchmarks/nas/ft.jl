# LIMITATION: cuNumeric has no NPB 46-bit RNG primitive, so exact initial
# conditions are generated on the host during each timed run and copied into a
# Legate array. The full 3-D FFT is a native Legate auto task and distributes
# across the available GPUs. Checksum sampling is currently expressed as a
# full masked reduction because cuNumeric has no indexed-reduction primitive.

struct CuNumericNASFTState{A,T,M,X,Y,Z,H,C}
    u0::A
    u1::A
    twiddle::T
    mask::M
    ix2::X
    iy2::Y
    iz2::Z
    host_initial::H
    checksums::C
end

function cunumeric_nas_ft_frequency_squares(n)
    return Float64[((i + n÷2) % n - n÷2)^2 for i in 0:(n - 1)]
end

function initialize(b::NASFourierTransform{Float64}; mod=cuNumeric)
    p = validate_nas_ft(b)
    shape = (p.nx, p.ny, p.nz)
    u0 = mod.zeros(ComplexF64, shape)
    u1 = mod.zeros(ComplexF64, shape)
    twiddle = mod.zeros(Float64, shape)
    mask = mod.NDArray(nas_ft_checksum_mask(p))
    ix2 = mod.NDArray(reshape(cunumeric_nas_ft_frequency_squares(p.nx), p.nx, 1, 1))
    iy2 = mod.NDArray(reshape(cunumeric_nas_ft_frequency_squares(p.ny), 1, p.ny, 1))
    iz2 = mod.NDArray(reshape(cunumeric_nas_ft_frequency_squares(p.nz), 1, 1, p.nz))
    host = Array{ComplexF64}(undef, shape)
    return (CuNumericNASFTState(u0, u1, twiddle, mask, ix2, iy2, iz2, host, Any[]),)
end

function run!(b::NASFourierTransform, s::CuNumericNASFTState)
    p = nas_ft_parameters(b.class)
    nas_ft_initial_conditions!(s.host_initial)
    copyto!(s.u0, s.host_initial)
    ap = -4.0*NAS_FT_ALPHA*pi^2
    s.twiddle .= exp.(ap .* (s.ix2 .+ s.iy2 .+ s.iz2))
    fft!(s.u0)
    empty!(s.checksums)
    for _ in 1:p.niter
        s.u0 .*= s.twiddle
        copyto!(s.u1, s.u0)
        ifft!(s.u1)
        push!(s.checksums, sum(s.u1 .* s.mask))
    end
    return s.checksums
end

function check_benchmark_correctness(
    b::NASFourierTransform, gs::GlobalSettings; mod=cuNumeric
)
    state = only(initialize(b; mod))
    got = ComplexF64[only(Array(x)) for x in run!(b, state)]
    return nas_ft_verified(b.class, got) ? "pass" : "fail"
end
