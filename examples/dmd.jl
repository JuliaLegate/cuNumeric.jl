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
    Ã = cuNumeric.transpose(U) * B

    E = eigen(Ã) # always complex, even for a real Ã
    Φ = cuNumeric.as_type(B, ComplexF32) * E.vectors

    return E.values, Φ
end

function main()
    
    N, M = 64, 80
    κ, dt = 0.01f0, 0.05f0
    xs = range(0f0, 1f0; length=N)
    times = Float32.(0:(M - 1)) .* dt

    # Two spatial modes of the heat equation with zero boundary values.
    ϕ1 = [sinpi(x) * sinpi(y) for x in xs, y in xs]
    ϕ2 = [sinpi(2x) * sinpi(3y) for x in xs, y in xs]
    σ1 = -2f0 * Float32(π)^2 * κ
    σ2 = -13f0 * Float32(π)^2 * κ

    # Each column is one flattened snapshot.
    snapshots = vec(ϕ1) * transpose(exp.(σ1 .* times)) +
                0.6f0 .* (vec(ϕ2) * transpose(exp.(σ2 .* times)))

    rank = 2
    λ, Φ = dmd(X, rank)

    vals = cuNumeric.@allowscalar Array(λ)
    order = sortperm(abs.(vals); rev=true)

    expected = exp.([σ1, σ2] .* dt)
    @show vals[order] expected
    @assert isapprox(vals[order], expected; rtol=1f-4, atol=1f-6)
end

main()
