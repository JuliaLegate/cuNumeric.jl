# Performance Tips


## Avoid Scalar Indexing
Accessing elements of an NDArray one at a time (e.g., `arr[5]`) is slow and should be avoided. Indexing like this requires data to be trasfered between device and host and maybe even communicated across nodes. Scalar indexing will emit an error which can be opted out of with `@allowscalar` or `allwoscalar() do ... end`. Several functions in the existing API invoke scalar indexing and are intended for testing (e.g., the `==` operator). 

## Avoid Implicit Promotion
Mixing integral types of difference size (e.g., `Float64` and `Float32`) will result in implicit promotion of the smaller type to the larger types. This creates a copy of the data and hurts performance. Implicit promotion from a smaller integral type to a larger integral type will emit an error which can be opted out of with `@allowpromotion` or `allowpromotion() do ... end`. This error is common when mixing literals with `NDArrays`. By default a floating point literal (i.e., 1.0) is `Float64` but the default type of an `NDArray` is `Float32`. 

## Kernel Fusion
cuPyNumeric does not fuse independent operations automatically. This is a priority for a future release.