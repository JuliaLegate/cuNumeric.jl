# Exact DMD of an N×M snapshot matrix (space × time). Rank is min(20, M-1),
# matching examples/dmd.jl.
#
# N is spatial degrees of freedom (rows of X), not a grid side length.
# The SVD is tall-skinny: X1 is N × (M-1). With M (and rank r) fixed, every
# term below is Θ(N) or O(1), so weak scaling is N ∝ P — not the N³ of a
# square SVD.

abstract type AbstractDMD{T} <: AbstractBenchmark{T} end

Base.@kwdef struct DMDBaseline{T} <: AbstractDMD{T}
    N::Int
    M::Int
end

Base.@kwdef struct DMDLifetimes{T} <: AbstractDMD{T}
    N::Int
    M::Int
end

name(::DMDBaseline) = "dmd_baseline"
name(::DMDLifetimes) = "dmd_lifetimes"
dims(b::AbstractDMD) = (b.N, b.M)
data(b::AbstractDMD{T}) where {T} = "DMD with T=$(T), N=$(b.N), M=$(b.M)"
allowed_types(::Type{<:AbstractDMD}) = cuNumeric.SUPPORTED_FLOAT_TYPES

function build_benchmark(::Type{A}, ::Type{T}, N, M) where {A<:AbstractDMD,T}
    return A{T}(; N=N, M=M)
end

_dmd_rank(b::AbstractDMD) = min(20, b.M - 1)

# X1 is m×n, m=N spatial points, n=M-1 snapshots, r = min(20, n).
#   2mn² + 11n³     thin SVD (Golub & Van Loan, Matrix Computations, 4ed, §8.6)
#   2mnr + mr       B = X2 V_r Σ_r^{-1}   (GEMM + column scale)
#   2mr²            Ã = U_r' B
#   25r³            eigen(Ã) (LAPACK xGEEV)
#   2mr²            Φ = B W
function total_flops(b::AbstractDMD)
    m = b.N
    n = b.M - 1
    r = _dmd_rank(b)
    return (
        2 * m * n^2 + 11 * n^3 +
        2 * m * n * r + m * r +
        2 * m * r^2 +
        25 * r^3 +
        2 * m * r^2
    )
end

function initialize(b::AbstractDMD{T}; mod=cuNumeric) where {T}
    # X1 is N×(M-1); the SVD backend requires m >= n.
    b.N >= b.M - 1 || throw(
        ArgumentError("DMD snapshot matrix is N×M with N ≥ M-1 (got N=$(b.N), M=$(b.M))")
    )
    X = mod.rand(T, b.N, b.M)
    GC.gc()
    return (X,)
end

_dmd_T(A) = A isa NDArray ? cuNumeric.transpose(A) : transpose(A)
_dmd_row(v) = v isa NDArray ? cuNumeric.reshape(v, (1, length(v))) : reshape(v, 1, length(v))

# svd / eigen return factorizations whose stores the lifetime rewriter cannot
# see, so those stay outside the macro. The GEMM lift is wrapped.
#
# Do not form Diagonal(1 ./ S) inside @analyze_lifetimes: the rewriter treats
# `1 ./ S` as a last-used temp and destroy!s it, while Diagonal still holds
# that same vector. Scale columns with a broadcast instead (same math).
function _dmd_factors(X, r)
    n = size(X, 2)
    X1 = X[:, 1:(n - 1)]
    X2 = X[:, 2:n]
    F = svd(X1)
    rk = min(r, length(F.S))
    return X2, F.U[:, 1:rk], F.Vt[1:rk, :], F.S[1:rk]
end

let body = quote
        Sinv = eltype(X)(1) ./ _dmd_row(S)
        B = (X2 * _dmd_T(Vt)) .* Sinv
        Ã = _dmd_T(U) * B
        (B, Ã)
    end
    @eval _dmd_project(::DMDBaseline, X, X2, U, Vt, S) = $body
    @eval _dmd_project(::DMDLifetimes, X, X2, U, Vt, S) = @analyze_lifetimes $body
end

function _dmd_compute!(b::AbstractDMD, X, r)
    X2, U, Vt, S = _dmd_factors(X, r)
    B, Ã = _dmd_project(b, X, X2, U, Vt, S)
    E = eigen(Ã)
    CT = Complex{eltype(X)}
    Bc = X isa NDArray ? cuNumeric.as_type(B, CT) : CT.(B)
    return E.values, Bc * E.vectors
end

run!(b::AbstractDMD, X) = _dmd_compute!(b, X, _dmd_rank(b))

correctness_supported(::AbstractDMD) = true

function check_benchmark_correctness(
    b::AbstractDMD{T}, gs::GlobalSettings; mod=cuNumeric, atol=1e-3, rtol=1e-3
) where {T}
    mod === cuNumeric || return "skipped"

    Xh = rand(T, b.N, b.M)
    X = NDArray(Xh)
    r = _dmd_rank(b)
    # Values, not lifetimes: compare against the baseline body on host and device.
    ref = DMDBaseline{T}(; N=b.N, M=b.M)
    λ, _ = _dmd_compute!(ref, X, r)
    λh, _ = _dmd_compute!(ref, Xh, r)

    mag = sort(abs.(Array(λ)); rev=true)
    magh = sort(abs.(λh); rev=true)
    return isapprox(mag, magh; atol=atol, rtol=rtol) ? "pass" : "fail"
end

register_benchmark("dmd_baseline", DMDBaseline)
register_benchmark("dmd_lifetimes", DMDLifetimes)
