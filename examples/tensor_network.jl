#= Contract a two-site tensor network, then optionally pull a host scalar.

The first contractions build an MPS-like two-site state and apply a Heisenberg
operator. Those results stay on device as NDArrays. Fully contracting to
⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩ goes through tensorscalar and therefore needs @allowscalar.
=#

using cuNumeric
using LinearAlgebra
using Random
using TensorOperations

Random.seed!(1234)

const PHYSICAL_DIM = 2
const BOND_DIM = 16

σx = ComplexF64[0 1; 1 0]
σy = ComplexF64[0 -im; im 0]
σz = ComplexF64[1 0; 0 -1]
Hmatrix = (kron(σx, σx) + kron(σy, σy) + kron(σz, σz)) / 4
H = NDArray(reshape(Hmatrix, PHYSICAL_DIM, PHYSICAL_DIM, PHYSICAL_DIM, PHYSICAL_DIM))

# MPS tensors A1[ap, s1, c], A2[c, s2, bp] and boundary environments.
A1 = NDArray(randn(ComplexF64, BOND_DIM, PHYSICAL_DIM, BOND_DIM))
A2 = NDArray(randn(ComplexF64, BOND_DIM, PHYSICAL_DIM, BOND_DIM))
L = NDArray(randn(ComplexF64, BOND_DIM, BOND_DIM))
R = NDArray(randn(ComplexF64, BOND_DIM, BOND_DIM))

# Open bonds remain, so both results are NDArrays.
@tensor ψ[a, s1, s2, b] :=
    L[a, ap] * A1[ap, s1, c] * A2[c, s2, bp] * R[bp, b]
@tensor Hψ[a, s1, s2, b] := H[s1, s2, t1, t2] * ψ[a, t1, t2, b]

println("ψ is a ", typeof(ψ), " of size ", size(ψ))
println("Hψ is a ", typeof(Hψ), " of size ", size(Hψ))

# Fully contracted @tensor results go through tensorscalar, which indexes the
# rank-zero NDArray. That is scalar indexing and needs @allowscalar.
energy, norm² = @allowscalar begin
    @tensor begin
        e = conj(ψ[a, s1, s2, b]) * Hψ[a, s1, s2, b]
        n = conj(ψ[a, s1, s2, b]) * ψ[a, s1, s2, b]
    end
    e, n
end

println("two-site energy = ", real(energy / norm²))
