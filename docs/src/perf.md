# Performance Tips

## Avoid Scalar Indexing
Accessing elements of an NDArray one at a time (e.g., `arr[5]`) is slow and should be avoided. Indexing like this requires data to be trasfered between device and host and maybe even communicated across nodes. Scalar indexing will emit an error which can be opted out of with `@allowscalar` or `allwoscalar() do ... end`. Several functions in the existing API invoke scalar indexing and are intended for testing (e.g., the `==` operator).

## Avoid Implicit Promotion
Mixing integral types of different size (e.g., `Float64` and `Float32`) will result in implicit promotion of the smaller type to the larger types. This creates a copy of the data and hurts performance. Implicit promotion from a smaller integral type to a larger integral type will emit an error which can be opted out of with `@allowpromotion` or `allowpromotion() do ... end`. This error is common when mixing literals with `NDArrays`. By default a floating point literal (i.e., 1.0) is `Float64` but the default type of an `NDArray` is `Float32`.

## Setting Hardware Configuration

There is no programatic way to set the hardware configuration used by CuPyNumeric (as of 26.01). By default, the hardware configuration is set automatically by Legate. This configuration can be manipulated through the following environment variables:

- `LEGATE_SHOW_CONFIG` : When set to 1, the Legate config is printed to stdout
- `LEGATE_AUTO_CONFIG`: When set to 1, Legate will automatically choose the hardware configuration
- `LEGATE_CONFIG`: A string representing the hardware configuration to set

These variables must be set before launching the Julia instance running cuNumeric.jl. We recommend setting `export LEGATE_SHOW_CONFIG=1` so that the hardware configuration will be printed when Legate starts. This output is automatically captured and relayed to the user.

To manually set the hardware configuration, `export LEGATE_AUTO_CONFIG=0`, and then define your own config with something like `export LEGATE_CONFIG="--gpus 1 --cpus 10 --ompthreads 10"`. We recommend using the default memory configuration for your machine and only settings the `gpus`, `cpus` and `ompthreads`. More details about the Legate configuration can be found in the [NVIDIA Legate documentation](https://docs.nvidia.com/legate/latest/usage.html#resource-allocation). If you know where Legate is installed on your computer you can also run `legate --help` for more detailed information.


## Kernel Fusion
cuPyNumeric does not fuse independent operations automatically, even in broadcast expressions. This is a priority for a future release.
