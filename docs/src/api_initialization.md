# Initialization

Constructors for new `NDArray`s. Default floating-point type is `Float32`.

## Basic Initialization

### Uninitialized arrays

Use `NDArray{T}(undef, dims...)` or `NDArray{T}(undef, dims::Tuple)` when the
next operation writes every element. `similar(A)` and `similar(A, T, dims)`
also return uninitialized arrays. Assign all elements before reading them;
use `cuNumeric.zeros` when the initial zero values are needed.

```julia
A = NDArray{Float32}(undef, 2, 3)
fill!(A, 1f0)
```

### zeros

```@docs
cuNumeric.zeros
```

### ones

```@docs
cuNumeric.ones
```

### fill

```@docs
cuNumeric.fill
```

### trues

```@docs
cuNumeric.trues
```

### falses

```@docs
cuNumeric.falses
```

## Special Matrices

### Diagonal

Construct a `Diagonal` matrix whose elements (diagonal only) are stored in an `NDArray`.
`LinearAlgebra.I` can be used to construct dense identity matrices as well.

```julia
using LinearAlgebra
using cuNumeric

D = Diagonal(cuNumeric.ones(Float32, 5))  # preferred for diagonal work
I32 = NDArray{Float32}(I, 5, 5)           # dense Float32 identity
Ib = NDArray(I, 5, 5)                     # Bool identity
```

See [Linear Algebra](./linalg.md#diagonal-and-identity) for preferred patterns.

## Random Numbers

### rand

```@docs
cuNumeric.rand
```

### rand!

```@docs
Random.rand!(::NDArray{<:cuNumeric.SUPPORTED_FLOAT_TYPES})
```

## randn

```@docs
cuNumeric.randn
```

## randn!

```@docs
Random.randn!(::NDArray{<:cuNumeric.SUPPORTED_FLOAT_TYPES})
```

## randexp

```@docs
cuNumeric.randexp
```

## randexp!

```@docs
Random.randexp!(::NDArray{<:cuNumeric.SUPPORTED_FLOAT_TYPES})
```

See [Random](./api_random.md) for BitGenerators (`XORWOW`, `MRG32k3a`,
`PHILOX4_32_10`), `Generator`, and `default_rng`.
