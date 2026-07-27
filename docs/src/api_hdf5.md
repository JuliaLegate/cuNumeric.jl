# HDF5

> [!NOTE]
> HDF5 support is planned. Signatures below are placeholders and will be replaced with `@docs` blocks when the API is implemented.

I/O helpers for reading and writing `NDArray`s via HDF5. Prefer these over host-side gather + HDF5.jl when arrays are large or distributed.

## h5read

```julia
# Planned:
# cuNumeric.h5read(path, dataset) -> NDArray
```

Load a dataset from an HDF5 file into an `NDArray`.

## h5write

```julia
# Planned:
# cuNumeric.h5write(path, dataset, arr::NDArray)
```

Write an `NDArray` to an HDF5 dataset.

## Related

- Example sketch: [HDF5 I/O](./examples/hdf5.md)
- Host conversion when you must leave the runtime: `Array(arr)` (see [NDArray Reference](./api.md))
