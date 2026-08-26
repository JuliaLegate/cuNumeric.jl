# Tensor Network Contraction

[TensorOperations.jl](https://quantumkithub.github.io/TensorOperations.jl/stable/)
provides Einstein-index notation through `@tensor`. Loading it together with
cuNumeric activates the `NDArray` extension, so contractions and their
intermediate tensors remain in Legate-managed memory.

This example applies the two-site spin-1/2 Heisenberg interaction

```math
H = S^x_1 S^x_2 + S^y_1 S^y_2 + S^z_1 S^z_2
```

to a tensor-network state ``\psi_{a s_1 s_2 b}``. The indices ``a`` and ``b``
are tensor-network bonds, while ``s_1`` and ``s_2`` are the two physical spin
indices. It then contracts every index to compute the normalized expectation
value ``\langle\psi|H|\psi\rangle/\langle\psi|\psi\rangle``.

```julia
# found in examples/tensor_network.jl
using cuNumeric
using LinearAlgebra
using Random
using TensorOperations

Random.seed!(1234)

physical_dim = 2
left_bond_dim = 16
right_bond_dim = 16

ψ = NDArray(
    randn(
        ComplexF64,
        left_bond_dim,
        physical_dim,
        physical_dim,
        right_bond_dim,
    ),
)

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
```

`Hψ` is an `NDArray`, while the fully contracted `energy` and `norm²` are Julia
scalars. TensorOperations releases temporary `NDArray`s created while lowering
larger tensor networks to pairwise contractions.

The extension also supports output-index permutations, traces, conjugation, and
accumulation into an existing tensor. See [Tensor contractions](../linalg.md#Tensor-contractions)
for the lower-level `contract` and `contract!` interfaces.
