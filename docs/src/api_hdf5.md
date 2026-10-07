# HDF5

`h5read` and `h5write` transfer datasets between HDF5 files and runtime-managed
`NDArray`s without gathering them into Julia `Array`s.

## Example

```julia
using cuNumeric

field = cuNumeric.fill(3.5f0, 128, 64)
cuNumeric.h5write("checkpoint.h5", "field", field)
cuNumeric.Legate.runtime_sync()

restored = cuNumeric.h5read("checkpoint.h5", "field"; layout=:row)
@assert size(restored) == (128, 64)
@assert eltype(restored) == Float32
@assert cuNumeric.compare(fill(3.5f0, 128, 64), restored, 0, 0)
```

`h5write` submits work to Legate and can return before the file write has completed.
Synchronize before accessing the file outside the runtime, moving or deleting it, or
exiting immediately after the write.

`h5write` removes a leftover empty or truncated `.h5` before launching the
write, which is what otherwise aborts `HDF5CombineVDS`. A valid HDF5 file is
left for Legate to overwrite. The `*_legate_vds` sidecar is left alone: after
a Legate write that directory is the data, and the `.h5` is only an index.
`h5read` rejects a missing path, a directory, or a file that does not start
with the HDF5 signature, instead of opening it in a Legate task.

## Dataset layout

```julia
row_major = cuNumeric.h5read("python.h5", "field")
column_major = cuNumeric.h5read("julia.h5", "field"; layout=:col)
```

`layout=:row` is the default for NumPy/h5py, cuPyNumeric, and `cuNumeric.h5write`.
Use `layout=:col` for multidimensional datasets written by HDF5.jl. One-dimensional
datasets are unaffected. Other keywords are forwarded to `Legate.h5read`.

Tests cover `Float32`, `Float64`, `Int32`, and `Int64` arrays with one to three
dimensions. Other types depend on the Legate HDF5 backend.

## API reference

### h5read

```@docs
cuNumeric.h5read
```

### h5write

```@docs
cuNumeric.h5write
```
