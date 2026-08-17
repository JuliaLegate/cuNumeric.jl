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

## Snapshots from Gray-Scott

The [Gray-Scott](./grayscott.md) example writes its `u` field to
`gray-scott.h5` every 20 steps, one flattened frame per column:

```julia
snapshots = cuNumeric.zeros(Float32, N * N, n_steps ÷ snapshot_interval)

# inside the time loop
if n%snapshot_interval == 0
    snapshots[:, n ÷ snapshot_interval] = cuNumeric.reshape(u, (N * N, 1))
end

cuNumeric.h5write(SNAPSHOT_FILE, "u", snapshots)
# h5write is asynchronous, so flush before another process opens the file.
cuNumeric.Legate.runtime_sync()
```

The snapshots never touch the host: they are assembled into a device array and
handed straight to [HDF5](../api_hdf5.md).

## Decomposition

Run `examples/gray-scott.jl` first to produce the file, then:

```julia
# found in examples/dmd.jl
using cuNumeric
using LinearAlgebra
using Printf

const SNAPSHOT_FILE = "gray-scott.h5"

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

X = cuNumeric.h5read(SNAPSHOT_FILE, "u")
n_points, n_snapshots = size(X)
N = isqrt(n_points)

λ, Φ = dmd(X, 20)

# Only r eigenvalues, so ranking them on the host costs nothing.
vals = Array(λ)
order = sortperm(abs.(vals); rev=true)

println("mode   |λ|      cycles/snapshot")
for i in order[1:min(5, end)]
    @printf("%4d   %6.4f   %+8.4f\n", i, abs(vals[i]), angle(vals[i]) / 2π)
end

# The slowest-decaying mode is the pattern the simulation settles into.
lead = order[1]
mode = cuNumeric.reshape(abs.(Φ[:, lead:lead]), (N, N))
cuNumeric.h5write("dmd-mode.h5", "leading", mode)
cuNumeric.Legate.runtime_sync()
```

On 100 snapshots of a ``100 \times 100`` grid this prints something like:

```
100 snapshots of 10000 points

mode   |λ|      cycles/snapshot
  17   0.9945    +0.0091
  18   0.9945    -0.0091
  14   0.9928    +0.0000
  19   0.9825    +0.0202
  20   0.9825    -0.0202
```

Every ``|\lambda|`` sits just below 1, which is the pattern settling rather than
growing, and the oscillatory modes come in the conjugate pairs a real operator
must produce.

## Notes

- `svd`, `eigen`, and matrix multiply all run on device; see
  [Linear Algebra](../linalg.md).
- `Diagonal(1.0f0 ./ F.S[1:r])` scales by ``\Sigma^{-1}`` without building a
  dense ``r \times r`` matrix. Note the `1.0f0` — an `Int` literal would try to
  widen the `Float32` singular values.
- `eigen` is always complex, so `B` is converted with `cuNumeric.as_type` before
  multiplying by the eigenvectors.
- The eigenvalues are only `r` numbers, so sorting and printing them on the host
  is cheap. The modes stay on device.
