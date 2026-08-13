# Random

Module-level [`rand`](@ref) and [`randn`](@ref) are also listed under
[Initialization](./api_initialization.md). This page covers the cuPyNumeric RNG
stack: BitGenerators, `Generator`, and `default_rng`.

Draws go through cuRAND with the [`XORWOW`](@ref) random number generator by default. We support `Float32` and `Float64` uniforms and normals; integers are signed `Int16` / `Int32` / `Int64` only. Ranged integers use Julia `rand(1:10, dims...)` (inclusive).

`Random.seed!` is not hooked. Use [`default_rng`](@ref) with an explicit seed
(or a specific BitGenerator) when you need a private stream. Module-level
`rand` / `randn` always use a process-global XORWOW generator.

## Module-level

```@docs
cuNumeric.rand
Random.rand!(::NDArray{<:AbstractFloat})
cuNumeric.randn
Random.randn!(::NDArray{<:AbstractFloat})
```

## BitGenerators

```@docs
cuNumeric.BitGenerator
cuNumeric.XORWOW
cuNumeric.MRG32k3a
cuNumeric.PHILOX4_32_10
```

## Generator

```@docs
cuNumeric.Generator
cuNumeric.default_rng
```

```julia
g = cuNumeric.default_rng()                          # XORWOW, fresh seed
g = cuNumeric.default_rng(1234)                      # XORWOW, fixed seed
g = cuNumeric.default_rng(cuNumeric.PHILOX4_32_10, 1)
A = cuNumeric.random(g, Float32, (8, 8))             # U[0, 1)
cuNumeric.randn!(g, A; loc=0, scale=1)               # N(loc, scale²)
```
