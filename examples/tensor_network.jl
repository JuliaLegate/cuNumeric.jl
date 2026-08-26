#= Apply a two-site Heisenberg interaction to a tensor-network state.

The open indices `a` and `b` are bond dimensions. The `s1` and `s2` indices
describe two spin-1/2 sites. TensorOperations chooses pairwise contractions and
cuNumeric keeps the tensors and intermediates in Legate-managed memory.
=#

using cuNumeric
using LinearAlgebra
using Random
using TensorOperations

Random.seed!(1234)

const PHYSICAL_DIM = 2
const LEFT_BOND_DIM = 16
const RIGHT_BOND_DIM = 16

# A random two-site tensor ψ[a, s1, s2, b].
ψ = NDArray(
    randn(
        ComplexF64,
        LEFT_BOND_DIM,
        PHYSICAL_DIM,
        PHYSICAL_DIM,
        RIGHT_BOND_DIM,
    ),
)

# H = S₁⋅S₂ for two spin-1/2 sites.
σx = ComplexF64[0 1; 1 0]
σy = ComplexF64[0 -im; im 0]
σz = ComplexF64[1 0; 0 -1]
Hmatrix = (kron(σx, σx) + kron(σy, σy) + kron(σz, σz)) / 4
H = NDArray(reshape(Hmatrix, 2, 2, 2, 2))

@tensor begin
    Hψ[a, s1, s2, b] := H[s1, s2, t1, t2] * ψ[a, t1, t2, b]
    energy = conj(ψ[a, s1, s2, b]) * Hψ[a, s1, s2, b]
    norm² = conj(ψ[a, s1, s2, b]) * ψ[a, s1, s2, b]
end

println("two-site energy = ", real(energy / norm²))
