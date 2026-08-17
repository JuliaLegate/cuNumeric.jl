# Spectral Poisson on a stack of periodic N×N grids.
# Each of the M right-hand sides is an independent ∇²u = f solve:
#   û = fft(f),  û *= −1/|k|²,  u = ifft(û)
# (electrostatics / gravity, not an FFT round-trip).
#
# The transform is over the last two axes, so the leading batch axis may
# partition across GPUs. A single all-axes 2-d FFT cannot.
# Weak scaling: M ∝ P, N fixed.

function _integer_fftfreq(n::Int)
    n2 = n ÷ 2
    return iseven(n) ? vcat(0:(n2 - 1), (-n2):-1) : vcat(0:n2, (-n2):-1)
end

function _poisson_inv_laplacian(::Type{T}, n::Int) where {T}
    freq = _integer_fftfreq(n)
    invk = Matrix{T}(undef, n, n)
    s = T(4 * π^2)
    for j in 1:n, i in 1:n
        k2 = s * T(freq[i]^2 + freq[j]^2)
        invk[i, j] = k2 == 0 ? zero(T) : -inv(k2)
    end
    return invk
end

Base.@kwdef struct PoissonFFT{T} <: AbstractBenchmark{T}
    N::Int
    M::Int
end

name(::PoissonFFT) = "poisson_fft"
dims(b::PoissonFFT) = (b.N, b.M)
data(b::PoissonFFT{T}) where {T} = "Poisson FFT with T=$(T), N=$(b.N), M=$(b.M)"
allowed_types(::Type{PoissonFFT}) = cuNumeric.SUPPORTED_FLOAT_TYPES

# 2-d C2C FFT: 5 n log2(n) real flops per 1-d line, 2n lines → 10 n² log2(n).
# Forward + inverse, plus n² complex muls (6 real flops each), times M batches.
function total_flops(b::PoissonFFT)
    n = b.N
    m = b.M
    n2 = n * n
    return m * (20 * n2 * log2(n) + 6 * n2)
end

function initialize(b::PoissonFFT{T}; mod=cuNumeric) where {T}
    f = mod.rand(T, b.M, b.N, b.N)
    CT = Complex{T}
    fc = f isa NDArray ? cuNumeric.as_type(f, CT) : CT.(f)
    work = copy(fc)
    kinv_h = _poisson_inv_laplacian(T, b.N)
    kinv = if fc isa NDArray
        reshape(NDArray(kinv_h), 1, b.N, b.N)
    else
        reshape(kinv_h, 1, b.N, b.N)
    end
    GC.gc()
    return fc, work, kinv
end

function run!(::PoissonFFT, fc, work, kinv)
    copyto!(work, fc)
    cuNumeric.batched_fft!(work)
    work .*= kinv
    cuNumeric.batched_ifft!(work)
    return work
end

correctness_supported(::PoissonFFT) = true

function check_benchmark_correctness(
    ::PoissonFFT{T}, gs::GlobalSettings; mod=cuNumeric, atol=1e-3, rtol=1e-3
) where {T}
    mod === cuNumeric || return "skipped"
    n = 32
    xs = range(T(0), T(1); length=n + 1)[1:(end - 1)]
    u_true = T[sin(2T(π) * x) * sin(2T(π) * y) for x in xs, y in xs]
    f_h = (-8 * T(π)^2) .* u_true
    f = NDArray(reshape(f_h, 1, n, n))
    kinv = reshape(NDArray(_poisson_inv_laplacian(T, n)), 1, n, n)
    u = real(cuNumeric.batched_ifft(cuNumeric.batched_fft(f) .* kinv))
    return isapprox(Array(u)[1, :, :], u_true; atol=atol, rtol=rtol) ? "pass" : "fail"
end

register_benchmark("poisson_fft", PoissonFFT)
