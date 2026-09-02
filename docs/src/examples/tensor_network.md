# Tensor Network Contraction

[TensorOperations.jl](https://quantumkithub.github.io/TensorOperations.jl/stable/)
provides Einstein-index notation through `@tensor`. Loading it together with
cuNumeric activates the `NDArray` extension, so contractions and their
intermediate tensors remain in Legate-managed memory.

The two-site spin-1/2 Heisenberg interaction is

```math
H = S^x_1 S^x_2 + S^y_1 S^y_2 + S^z_1 S^z_2.
```

The examples below apply it to an MPS-like state with tensors ``A^{(1)}``,
``A^{(2)}`` and boundary environments ``L``, ``R``. The first contractions keep
open bond indices and stay `NDArray`s. The second contracts every index; the
ratio stays on device until `unwrap` for printing.

```julia
# found in examples/tensor_network.jl
using cuNumeric
using LinearAlgebra
using Random
using TensorOperations

Random.seed!(1234) # for Random.randn, cuNumeric expects seed via default_rng

physical_dim = 2
bond_dim = 16

σx = ComplexF64[0 1; 1 0]
σy = ComplexF64[0 -im; im 0]
σz = ComplexF64[1 0; 0 -1]
Hmatrix = (kron(σx, σx) + kron(σy, σy) + kron(σz, σz)) / 4
H = NDArray(reshape(Hmatrix, 2, 2, 2, 2))

A1 = NDArray(randn(ComplexF64, bond_dim, physical_dim, bond_dim))
A2 = NDArray(randn(ComplexF64, bond_dim, physical_dim, bond_dim))
L = NDArray(randn(ComplexF64, bond_dim, bond_dim))
R = NDArray(randn(ComplexF64, bond_dim, bond_dim))
```

## Apply a two-site operator

These contractions have free indices ``a, s_1, s_2, b``, so the results are
`NDArray`s. TensorOperations lowers the network to pairwise contractions and
releases the intermediate `NDArray`s.

```julia
@tensor ψ[a, s1, s2, b] :=
    L[a, ap] * A1[ap, s1, c] * A2[c, s2, bp] * R[bp, b]
@tensor Hψ[a, s1, s2, b] := H[s1, s2, t1, t2] * ψ[a, t1, t2, b]

println("ψ is a ", typeof(ψ), " of size ", size(ψ))
println("Hψ is a ", typeof(Hψ), " of size ", size(Hψ))
```

## Expectation value

A fully contracted `@tensor` assignment is a 0D `NDArray`, like `sum`.
Prefer to keep it that way and do device arithmetic (`./`) until you
need a host `Number`. `unwrap` (here, only to print) **blocks**. See
[Scalars](../api_tensor.md#Scalars).

```julia
@tensor energy = conj(ψ[a, s1, s2, b]) * Hψ[a, s1, s2, b]
@tensor norm² = conj(ψ[a, s1, s2, b]) * ψ[a, s1, s2, b]

println("two-site energy = ", real(unwrap(energy ./ norm²)))
```
See [Tensor Contractions](../api_tensor.md) documentation for more details.
