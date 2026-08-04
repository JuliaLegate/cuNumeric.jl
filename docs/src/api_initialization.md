# Initialization

Constructors for new `NDArray`s. Default floating-point type is `Float32`.

## zeros

```@docs
cuNumeric.zeros
```

## ones

```@docs
cuNumeric.ones
```

## fill

```@docs
cuNumeric.fill
```

## trues

```@docs
cuNumeric.trues
```

## falses

```@docs
cuNumeric.falses
```

## Identity

There is no NumPy-style `eye`. Prefer `Diagonal` for diagonal scaling and
`LinearAlgebra.I` for UniformScaling ops (`A + I`, `D + I`, …). Densify only
when a full identity matrix is required:

```julia
using LinearAlgebra
using cuNumeric

D = Diagonal(cuNumeric.ones(Float32, 5))  # preferred for diagonal work
I32 = NDArray{Float32}(I, 5, 5)           # dense Float32 identity
Ib = NDArray(I, 5, 5)                     # Bool identity (same as Matrix(I, ...))
```

See [Linear Algebra](./linalg.md#diagonal-and-identity) for preferred patterns.

## rand

```@docs
cuNumeric.rand
```

## rand!

```@docs
Random.rand!(::NDArray{Float64})
```

The backend currently draws `Float64` uniforms. `cuNumeric.rand(Float32, dims...)` converts for you. `rand!` on `NDArray` currently requires `Float64` storage.
