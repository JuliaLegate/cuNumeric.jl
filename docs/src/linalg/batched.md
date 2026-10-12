# Batched operations

The batched linear algebra functions apply an operation to every matrix in a
stack. They take exactly one leading batch dimension: `A` has shape
`(b, m, m)`, where `b` is the number of matrices. Results are `NDArray`s
or tuples of `NDArray`s, rather than factorization objects.

## Available functions

These are all currently available batched linear algebra entry points:

| Function | Operation | Result |
| :--- | :--- | :--- |
| `cuNumeric.batched_solve(A, B)` | Solve each system `A[i, :, :] * X[i, :, :] = B[i, :, :]` | Array of shape `(b, m, n)`, matching `B` |
| `cuNumeric.batched_cholesky(A)` | Compute each lower Cholesky factor | Array of shape `(b, m, m)`, with zeroed upper triangles |
| `cuNumeric.batched_eigen(A)` | Compute eigenvalues and right eigenvectors | `(values, vectors)` with shapes `(b, m)` and `(b, m, m)` |
| `cuNumeric.batched_eigvals(A)` | Compute eigenvalues only | Array of shape `(b, m)` |

For `batched_solve`, `B` must have shape `(b, m, n)`. Use `(b, m, 1)`
for one right-hand side per system. The batch sizes must match.

Cholesky requires positive-definite matrices. Eigenvalues and eigenvectors
are always complex, including for real input. The caveats under
[Cholesky](factorizations.md#Cholesky) and
[Eigendecomposition](factorizations.md#Eigendecomposition) also apply here.
See the [overview](../linalg.md#Supported-element-types) for accepted element
types and promotion rules.

## Example

Construct positive-definite matrices on the host, then transfer the stack to
an `NDArray`. All four operations below use the same input batch:

```julia
using cuNumeric, LinearAlgebra, Random

batch_size, m = 8, 32
A_host = Array{Float32}(undef, batch_size, m, m)
for i in 1:batch_size
    M = Random.randn(Float32, m, m)
    A_host[i, :, :] = M * M' + 32f0 * I
end

As = NDArray(A_host)
Bs = cuNumeric.rand(Float32, batch_size, m, 2)

Xs = cuNumeric.batched_solve(As, Bs)          # (8, 32, 2)
Ls = cuNumeric.batched_cholesky(As)           # (8, 32, 32)
values, vectors = cuNumeric.batched_eigen(As) # (8, 32), (8, 32, 32)
values_only = cuNumeric.batched_eigvals(As)   # (8, 32)
```

## Limits

Each matrix is factored on a single processor, so this is a good fit for many
small matrices. Each matrix must fit on one processor, and two or more batch
dimensions are not supported.

There are no batched SVD or QR entry points, no separate `batched_eigvecs`,
and no in-place `!` variants of the four linear algebra functions above.
Use `batched_eigen` when you need eigenvectors.

## Batched FFTs

The package also provides `batched_fft`, `batched_ifft`, `batched_fft!`,
and `batched_ifft!`. These transform the trailing dimensions while treating
dimension 1 as the batch. The in-place forms require complex arrays.
See [Batched transforms](../fft.md#Batched-transforms) for usage.
