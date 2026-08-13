# Random

Module-level [`rand`](@ref cuNumeric.rand) and [`randn`](@ref cuNumeric.randn)
are listed under [Initialization](./api_initialization.md). This page covers
the cuPyNumeric RNG stack: BitGenerators, `Generator`, and `default_rng`.

Draws go through cuRAND with the [`XORWOW`](@ref cuNumeric.XORWOW) random
number generator by default. We support `Float32` and `Float64` uniforms and
normals, `Bool` coin flips, and signed `Int16` / `Int32` / `Int64`. There is no native Bool distribution, so `rand(Bool, …)` draws `Int16` values in `{0,1}` and compares them to zero.

`Random.seed!` is not hooked. Use [`default_rng`](@ref cuNumeric.default_rng)
with an explicit seed (or a specific BitGenerator) when you need a private
stream. Module-level `rand` / `randn` always use a process-global XORWOW
generator.

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
