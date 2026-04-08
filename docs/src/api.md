
# Public API

User facing functions supported by cuNumeric.jl

```@contents
Pages = ["api.md"]
Depth = 2:2
```

Supported Unary Operations
===========================

The following unary operations are supported and can be broadcast over `NDArray`:

  - `-` (negation)
  - `!` (logical not)
  - `abs`
  - `acos`
  - `acosh`
  - `asin`
  - `asinh`
  - `atan`
  - `atanh`
  - `cbrt`
  - `cos`
  - `cosh`
  - `deg2rad`
  - `exp`
  - `exp2`
  - `expm1`
  - `floor`
  - `isfinite`
  - `log`
  - `log10`
  - `log1p`
  - `log2`
  - `rad2deg`
  - `sign`
  - `signbit`
  - `sin`
  - `sinh`
  - `sqrt`
  - `tan`
  - `tanh`
  - `^2`
  - `^-1` or `inv`

Differences
-----------
- The `acosh` function in Julia will error on inputs outside of the domain (x >= 1)
    but cuNumeric.jl will return NaN.

Examples
--------

```julia
A = cuNumeric.ones(Float32, 3, 3)

abs.(A)
log.(A .+ 1)
-sqrt.(abs.(A))
```


Supported Binary Operations
===========================

The following binary operations are supported and can be applied elementwise to pairs of `NDArray` values:

  • `+`
  • `-`
  • `*`
  • `/`
  • `^`
  • `<`
  • `<=`
  • `>`
  • `>=`
  • `==`
  • `!=`
  • `atan`
  • `hypot`
  • `max`
  • `min`
  • `lcm`
  • `gcd`

These operations are applied elementwise by default and follow standard Julia semantics.

Examples
--------

```julia
A = NDArray(randn(Float64, 4))
B = NDArray(randn(Float64, 4))

A + B
A / B
hypot.(A, B)
div.(A, B)
A .^ 2
```

```@autodocs
Modules = [cuNumeric]
Pages = ["ndarray/ndarray.jl", "ndarray/unary.jl", "ndarray/binary.jl", "cuNumeric.jl", "warnings.jl", "util.jl", "memory.jl", "scoping.jl"]
```

# CUDA.jl Tasking
This section will detail how to use custom CUDA.jl kernels with the Legate runtime. This is still a work in progress

```@autodocs
Modules = [cuNumeric]
Pages = ["cuda.jl"]
```

# CNPreferences

This section details how to set custom build configuration options. To see more details visit our install guide [here](./install.md).

```@autodocs
Modules = [CNPreferences]
Pages = ["CNPreferences.jl"]
```

# Internal API

```@autodocs
Modules = [cuNumeric]
Pages = ["ndarray/detail/ndarray.jl"]
```
