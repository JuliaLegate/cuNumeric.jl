# Dynamic Mode Decomposition

Dynamic mode decomposition takes a sequence of states from a simulation or an
experiment and finds the best linear operator that advances one state to the
next:

```math
x_{k+1} \approx A x_k
```

Its eigenvectors are spatial patterns and its eigenvalues say what each pattern
does per step: ``|\lambda|`` is growth or decay and ``\arg(\lambda)`` is
rotation. For a field on an ``N \times N`` grid, ``A`` is ``N^2 \times N^2`` and
far too large to form. DMD never does. Stack the states as columns of
``X = [x_1 \; \dots \; x_n]``, split it into

```math
X_1 = [x_1 \; \dots \; x_{n-1}], \qquad X_2 = [x_2 \; \dots \; x_n]
```

and take the rank-``r`` SVD ``X_1 = U \Sigma V^*``. Projecting ``A`` onto the
columns of ``U`` gives an ``r \times r`` matrix whose eigendecomposition carries
the dynamics:

```math
\tilde{A} = U^* X_2 V \Sigma^{-1}, \qquad
\Phi = X_2 V \Sigma^{-1} W
```

where ``W`` holds the eigenvectors of ``\tilde{A}``. The columns of ``\Phi`` are
the exact DMD modes, back in the full ``N^2``-dimensional space.

## Generate Input

To begin we will generate input from a temperature field containing two spatial patterns that decay at different rates.

```julia
using cuNumeric

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

cuNumeric.h5write("dmd-snapshots.h5", "u", NDArray(snapshots))

# In practice you do not write then immeaditely read 
# a file. In this example you do, so sync the runtime to
# ensure that the file is flushed. 
cuNumeric.Legate.runtime_sync()
```

## Decomposition

Run `examples/gray-scott.jl` first to produce the file, then:

```julia
# found in examples/dmd.jl
using cuNumeric
using LinearAlgebra

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

X = cuNumeric.h5read("dmd-snapshots.h5", "u")
rank = 2
λ, Φ = dmd(X, rank)

vals = cuNumeric.@allowscalar Array(λ)
order = sortperm(abs.(vals); rev=true)

expected = exp.([σ1, σ2] .* dt)
@show vals[order] expected
@assert isapprox(vals[order], expected; rtol=1f-4, atol=1f-6)
```

## Notes

- `eigen` is always complex, so `B` is converted with `cuNumeric.as_type` before
  multiplying by the eigenvectors.

