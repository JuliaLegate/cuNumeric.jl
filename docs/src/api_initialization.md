# Initialization

Constructors for new `NDArray`s. Default floating-point type is `Float32`.

## Basic Initialization

### Uninitialized arrays

User-facing uninitialized array constructors, such as `NDArray{T}(undef, dims...)`,
`NDArray{T}(undef, dims::Tuple)`, and ordinary `similar` calls, zero-initialize
storage so that the result can immediately be displayed or converted with `Array`.

```julia
A = NDArray{Float32}(undef, 2, 3)
fill!(A, 1f0)
copyto!(A, rand(Float32, 2, 3))   # or fill from a host Array
```

### From host data

`NDArray(host)` makes a new array from a Julia `Array`. `copyto!(A, host)`
refills an existing one without allocating.

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
