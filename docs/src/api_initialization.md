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

## eye

```@docs
cuNumeric.eye
```

## rand

```@docs
cuNumeric.rand
```

## rand!

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
