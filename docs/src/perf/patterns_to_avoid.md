# Patterns to Avoid

## Scalar indexing

Accessing elements of an NDArray one at a time (e.g., `arr[5]`) is slow and should be avoided. Indexing like this requires data to be transferred between device and host and maybe even communicated across nodes. Scalar indexing will emit an error which can be opted out of with `@allowscalar` or `allowscalar() do ... end`. Several functions in the existing API invoke scalar indexing and are intended for testing (e.g., the `==` operator).

`fetch` retrieves a native Julia scalar from a `CNScalar` or a single-element
`NDArray`, waiting for the result as needed. Prefer a native Julia number when
you already have one, and leave reductions / fully contracted `@tensor` results
in runtime-managed storage until you need the host value.
See [Scalars](../api_tensor.md#Scalars).

## Implicit promotion

Mixing integral types of different size (e.g., `Float64` and `Float32`) will result in implicit promotion of the smaller type to the larger types. This creates a copy of the data and hurts performance. Implicit promotion from a smaller integral type to a larger integral type will emit an error which can be opted out of with `@allowpromotion` or `allowpromotion() do ... end`. This error is common when mixing literals with `NDArrays`. By default a floating point literal (i.e., 1.0) is `Float64` but the default type of an `NDArray` is `Float32`.
