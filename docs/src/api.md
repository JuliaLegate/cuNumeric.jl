
# Public API

User facing functions supported by cuNumeric.jl

```@contents
Pages = ["api.md"]
Depth = 2:2
```

```@autodocs
Modules = [cuNumeric]
Pages = ["cuNumeric.jl", "ndarry.jl", "binary.jl", "unary.jl"]
```

## Initializing NDArrays

The CuPyNumeric C++ API only supports generating Float64 random numbers. The example below shows how you can get Float32 random numbers by casting. We plan to make this easier through `Base.convert` or by getting Float32 generating added to CuPyNumeric.

```julia
arr_fp64 = rand(NDArray, 100)
arr_fp32 = cuNumeric.as_type(arr_fp64, LegateType(Float32))
```

#### Methods to intiailize NDArrays

- `cuNumeric.zeros`
- `cuNumeric.full`
- `Random.rand!`
- `Random.rand`


## Slicing NDArrays
TODO

## Linear Algebra Operations

Matrix multiplicaiton is only implemented through `mul!`. Calling the `*` operator on a pair of 2D NDArrays will perform elementwise multiplication.

#### Implemented Linear Algebra Operations
- `LinearAlgebra.mul!`

## Unary Operations

All unary operations will return a new NDArray and are broadcast over an NDarray even without the `.` broadcasting syntax. In the current state, `.` broadcasting syntax will not work (e.g. `sin.(arr)`).

Unary operators can also be called with `map`. For example
```julia
arr = cuNumeric.rand(NDArray, 100)

res1 = sqrt(arr)
res2 = map(sqrt, arr)
```

#### Implemented Unary Operations
- `Base.abs`
- `Base.acos`
- `Base.asin`
- `Base.asinh`
- `Base.atanh`
- `Base.cbrt`
- `Base.conj`
- `Base.cos`
- `Base.cosh`
- `Base.deg2rad`
- `Base.exp`
- `Base.expm1`
- `Base.floor`
- `Base.log`
- `Base.log10`
- `Base.log1p`
- `Base.log2`
- `Base.:(-)`
- `Base.rad2deg`
- `Base.sin`
- `Base.sinh`
- `Base.sqrt`
- `Base.tan`
- `Base.tanh`

## Unary Reductions
Unary reductions convert an NDArray to a single number. Unary reductions cannot be called with `Base.reduce` at this time.

#### Implemented Unary Reductions

- `Base.all`
- `Base.any`
- `Base.maximum`
- `Base.minimum`
- `Base.prod`
- `Base.sum`

## Binary Operations

All binary operations will return a new NDArray and are broadcast over an NDarray even without the `.` broadcasting syntax. In the current state, `.` broadcasting syntax will not work (e.g. `sin.(arr)`).

Binary operators can also be called with `map`. For example
```julia
arr1 = cuNumeric.rand(NDArray, 100)
arr2 = cuNumeric.rand(NDArray, 100)

res1 = arr1*arr2
res2 = map(*, arr1, arr2)
```

#### Implemented Binary Operations
- `Base.:(+)`
- `Base.atan`
- `Base.:(/)`
- `Base.:(^)`
- `Base.div`
- `Base.hypot`
- `Base.:(*)`
- `Base.(-)`


## Timing cuNumeric.jl Code

These timers will block until all prior Legate operations are complete.

- `get_time_microseconds`
- `get_time_nanoseconds`
