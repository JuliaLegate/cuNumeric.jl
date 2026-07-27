# HDF5 I/O

> [!NOTE]
> HDF5 support is planned. This page is a placeholder for an end-to-end example once the API lands.

Reading and writing `NDArray`s through HDF5 will let you checkpoint distributed arrays and exchange data with NumPy / cuPyNumeric workflows without gathering everything to the host first.

## Planned sketch

```julia
using cuNumeric

# Write (API names TBD)
# cuNumeric.h5write("checkpoint.h5", "fields/u", u)

# Read into an NDArray (API names TBD)
# u = cuNumeric.h5read("checkpoint.h5", "fields/u")
```

When available, prefer the cuNumeric HDF5 entry points over collecting to a Julia `Array` and using HDF5.jl alone, so large arrays can stay partitioned across devices.

See [HDF5](../api_hdf5.md) in the Public API for the (forthcoming) function reference.
