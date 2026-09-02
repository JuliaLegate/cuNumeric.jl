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
    f = rand_array(mod, T, b.M, b.N, b.N)
    fc = astype_array(f, Complex{T})
    work = copy(fc)
    kinv = to_backend(mod, reshape(_poisson_inv_laplacian(T, b.N), 1, b.N, b.N))
    GC.gc()
    return fc, work, kinv
end

_trailing_fft_dims(A) = ntuple(i -> i + 1, ndims(A) - 1)

_batched_fft!(A::NDArray) = (cuNumeric.batched_fft!(A); A)
_batched_ifft!(A::NDArray) = (cuNumeric.batched_ifft!(A); A)
_batched_fft!(A) = (fft!(A, _trailing_fft_dims(A)); A)
_batched_ifft!(A) = (ifft!(A, _trailing_fft_dims(A)); A)

function run!(::PoissonFFT, fc, work, kinv)
    copyto!(work, fc)
    _batched_fft!(work)
    work .*= kinv
    _batched_ifft!(work)
    return work
end

correctness_problem(b::PoissonFFT{T}) where {T} = PoissonFFT{T}(; N=min(b.N, 32), M=1)

register_benchmark("poisson_fft", PoissonFFT)
