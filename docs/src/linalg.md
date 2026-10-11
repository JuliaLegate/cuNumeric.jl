# Linear Algebra

cuNumeric.jl provides matrix multiplication, solves, Cholesky, eigen, SVD, QR,
and related helpers for `NDArray`. For Einstein-index contractions, see
[Tensor Contractions](./api_tensor.md).

## Supported element types

All of the decompositions accept `Float32`, `Float64`, `ComplexF32`, and
`ComplexF64`. Integer and `Bool` inputs are converted to `Float64`. As
everywhere else in the package, that conversion needs `@allowpromotion` (or
`allowpromotion`) only when it widens the element type, so `Int64` and `UInt64`
pass through silently while `Int32`, smaller integers, and `Bool` do not.

## In this section

| Page | Topics |
| :--- | :--- |
| [Standard functions](linalg/operations.md) | Matrix and vector products, norms, linear systems, and helpers |
| [Factorizations](linalg/factorizations.md) | Cholesky, eigendecomposition, SVD, and QR |
| [Batched operations](linalg/batched.md) | Solves and decompositions for stacks of matrices |
| [Diagonal and identity](linalg/diagonal.md) | Structured diagonal matrices and uniform scaling |
| [Tuning](linalg/tuning.md) | Multi-GPU selection rules, cutoffs, and block sizes |

## Which entry point to use

Single matrices use familiar `LinearAlgebra` functions. Most decompositions
return standard factorization objects; QR returns cuNumeric's `NDArrayQR`. Stacks of matrices have no `LinearAlgebra`
equivalent, so they get their own `batched_*` names.

| Single matrix (2D) | Returns | Stack of matrices (3D) |
| :--- | :--- | :--- |
| `LinearAlgebra.cholesky(A)` | `Cholesky` | `cuNumeric.batched_cholesky(A)` |
| `LinearAlgebra.eigen(A)` | `Eigen` | `cuNumeric.batched_eigen(A)` |
| `LinearAlgebra.eigvals(A)` | `NDArray` | `cuNumeric.batched_eigvals(A)` |
| `LinearAlgebra.svd(A)` | `SVD` | not supported |
| `LinearAlgebra.qr(A)` | `NDArrayQR` | not supported |
| `cuNumeric.solve(A, b)`, `A \ b` | `NDArray` | `cuNumeric.batched_solve(A, B)` |

## Not available yet

There is no public dense-matrix `lu`, matrix `inv`, or `ldiv!` yet (beyond the
[diagonal-matrix operations](linalg/diagonal.md)). Elementwise `inv` / `^-1` are unary
operations, not matrix inverse.

Also missing: `eigh` / Hermitian eigen (needs `Hermitian` and `Symmetric`
support on `NDArray`), and batched SVD and QR.

## Reference

```@autodocs
Modules = [cuNumeric]
Pages = ["ndarray/linalg.jl", "ndarray/batched_linalg.jl", "ndarray/vector_linalg.jl", "ndarray/diagonal.jl"]
```
