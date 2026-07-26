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
Random.rand!(::NDArray{Float64})
```

The backend currently draws `Float64` uniforms. `cuNumeric.rand(Float32, dims...)` converts for you. `rand!` on `NDArray` currently requires `Float64` storage.
