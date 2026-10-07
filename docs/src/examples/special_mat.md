# Special Matrices

Currently cuNumeric only supports `LinearAlgebra.Diagonal`. Other special matrix types like `Tridiagonal` and `Symmetric` will follow.

Diagonal matrices are common, require only storage of the diagonal elements and are often simple to compute operations on (i.e. `LinearAlgebra.inv`). `LinearAlgebra.Diagonal` matrices can be constructed from 1D or 2D `NDArray`s and have certain operations implemented (i.e., `eigen` and `inv`).


```julia
using cuNumeric
using LinearAlgebra

one_dim = cuNumeric.NDArray([1,2,3,4,5])
two_dim = cuNumeric.rand(5,5)

D1 = Diagonal(one_dim)
D2 = Diagonal(two_dim)

evals, evecs = eigen(D1)
D1_inv = inv(D1)

D1 ./= 2 # stays diagonal
arr = D2 .+ two_dim # densifies because `two_dim` is not guranteed to be diagonal
```
