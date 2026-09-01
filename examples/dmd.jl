#= Dynamic mode decomposition of the Gray-Scott snapshots.

Reads gray-scott.h5 from the working directory if examples/gray-scott.jl has been
run, otherwise the smaller sample committed under examples/data (a 64x64 grid
sampled every 100 of 3000 steps).

DMD fits the best linear operator A with x_{k+1} ≈ A x_k across the snapshots.
A itself is N²×N² and never formed: the SVD of the snapshot matrix gives a rank-r
subspace, A is projected onto it, and the eigendecomposition of that small r×r
matrix carries the dynamics. Each eigenvalue λ says what its spatial mode does
from one snapshot to the next — |λ| is growth or decay, arg(λ) is rotation.
=#

using cuNumeric
using LinearAlgebra
using Printf

const SNAPSHOT_FILE = "gray-scott.h5"
const SAMPLE_FILE = joinpath(@__DIR__, "data", "gray-scott.h5")

"""
    dmd(X, r)

Eigenvalues and exact DMD modes of the snapshot matrix `X`, truncated to rank `r`.
Columns of `X` are successive states.
"""
function dmd(X::NDArray{Float32,2}, r::Int)
    n = size(X, 2)
    X1 = X[:, 1:(n - 1)] # states x_1 … x_{n-1}
    X2 = X[:, 2:n]       # the same states advanced one snapshot

    F = svd(X1)
    r = min(r, length(F.S))
    U = F.U[:, 1:r]
    Vt = F.Vt[1:r, :]

    # Σ⁻¹ stays a Diagonal instead of a dense r×r matrix.
    Sinv = Diagonal(1.0f0 ./ F.S[1:r])

    # X2 V Σ⁻¹ appears in both the projected operator and the exact modes.
    B = X2 * cuNumeric.transpose(Vt) * Sinv
    A_tilde = cuNumeric.transpose(U) * B

    E = eigen(A_tilde) # always complex, even for a real Ã
    Φ = cuNumeric.as_type(B, ComplexF32) * E.vectors

    return E.values, Φ
end

function main()
    path = isfile(SNAPSHOT_FILE) ? SNAPSHOT_FILE : SAMPLE_FILE
    println("reading $path")

    X = cuNumeric.h5read(path, "u")
    n_points, n_snapshots = size(X)
    N = isqrt(n_points)
    println("$n_snapshots snapshots of $n_points points")

    λ, Φ = dmd(X, 20)

    # Only r eigenvalues, so ranking them on the host costs nothing.
    vals = Array(λ)
    order = sortperm(abs.(vals); rev=true)

    println("\nmode   |λ|      cycles/snapshot")
    for i in order[1:min(5, end)]
        @printf("%4d   %6.4f   %+8.4f\n", i, abs(vals[i]), angle(vals[i]) / 2π)
    end

    # The slowest-decaying mode is the pattern the simulation settles into.
    lead = order[1]
    mode = cuNumeric.reshape(abs.(Φ[:, lead:lead]), (N, N))
    cuNumeric.h5write("dmd-mode.h5", "leading", mode)
    cuNumeric.Legate.runtime_sync()
    println("\nwrote leading mode $lead to dmd-mode.h5")

    return λ, Φ
end

λ, Φ = main()
