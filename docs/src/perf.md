# Performance Tips


## Avoid Scalar Indexing
Accessing elements of an [NDArray](@ref) one at a time (e.g., `arr[5]`) is slow and should be avoided. Indexing like this requires data to be trasfered between device and host and maybe even communicated across nodes. In the future, scalar indexing will emit a warning which can be opted out of. Several functions in the existing API invoke scalar indexing and are intended for testing (e.g., the `==` operator). 

## Kernel Fusion
cuPyNumeric does not fuse independent operations automatically. This is a priority for the beta release.