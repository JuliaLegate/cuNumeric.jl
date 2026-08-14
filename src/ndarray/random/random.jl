#= Copyright 2026 Northwestern University,
 *                   Carnegie Mellon University University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author(s): David Krasowska <krasow@u.northwestern.edu>
 *            Ethan Meitz <emeitz@andrew.cmu.edu>
 *            Nader Rahhal <naderrahhal2026@u.northwestern.edu>
=#

# Module-level convenience API, mirroring cupynumeric.random._random.py.
# All draws go through get_static_generator().

@doc"""
    Random.rand!(arr::NDArray{<:SUPPORTED_FLOAT_TYPES})
    Random.rand!(arr::NDArray{<:SUPPORTED_COMPLEX_TYPES})
    Random.rand!(arr::NDArray{<:Union{Int16,Int32,Int64}})
    Random.rand!(arr::NDArray{Bool})

Fill `arr` in-place with uniform random values.

Floating arrays are filled from `[0, 1)`. Complex arrays draw independent
real/imag uniforms in `[0, 1)` (the unit square). Integer arrays use the full
range of the element type, matching Julia `rand(T)`. `Bool` arrays are fair
coin flips.
"""
function Random.rand!(arr::NDArray{<:SUPPORTED_FLOAT_TYPES})
    return random!(get_static_generator(), arr)
end

function Random.rand!(arr::NDArray{<:SUPPORTED_COMPLEX_TYPES})
    return random!(get_static_generator(), arr)
end

function Random.rand!(arr::NDArray{T}) where {T<:_RNG_INT_TYPES}
    return integers!(get_static_generator(), arr; low=typemin(T), high=typemax(T))
end

function Random.rand!(arr::NDArray{Bool})
    return random!(get_static_generator(), arr)
end

function Random.rand!(arr::NDArray{T}) where {T}
    return error(
        "rand! supports Float32, Float64, ComplexF32, ComplexF64, Bool, Int16, Int32, and Int64 NDArray storage"
    )
end

@doc"""
    cuNumeric.rand([T=Float32,] dims::Int...)
    cuNumeric.rand([T=Float32,] dims::Tuple)
    cuNumeric.rand(r::AbstractUnitRange, dims...)

Create a new `NDArray` filled with uniform random values.

Floating types (`Float32` / `Float64`) are drawn natively in `[0, 1)`.
`ComplexF32` / `ComplexF64` draw independent real/imag uniforms (unit square).
`Bool` is a fair coin flip. Integer types `Int16`, `Int32`, and `Int64` use the
full range of `T`. A unit range (`1:10`) draws inclusive integers, matching
Julia `rand(1:10, dims...)`.

Uses a process-global [`XORWOW`](@ref cuNumeric.XORWOW) generator. For a different engine or
seed, see [`default_rng`](@ref cuNumeric.default_rng) and [`Generator`](@ref cuNumeric.Generator).

# Examples
```@repl
cuNumeric.rand(2, 2)
cuNumeric.rand((4, 1))
cuNumeric.rand(0:9, 4, 4)
cuNumeric.rand(Bool, 8)
A = cuNumeric.zeros(Float32, 2, 2); cuNumeric.rand!(A)
```
"""
function rand(::Type{T}, dims::Dims) where {T<:SUPPORTED_FLOAT_TYPES}
    return random(get_static_generator(), T, dims)
end

function rand(::Type{T}, dims::Dims) where {T<:SUPPORTED_COMPLEX_TYPES}
    return random(get_static_generator(), T, dims)
end

function rand(::Type{T}, dims::Dims) where {T<:_RNG_INT_TYPES}
    return integers(get_static_generator(), T, dims; low=typemin(T), high=typemax(T))
end

rand(::Type{Bool}, dims::Dims) = random(get_static_generator(), Bool, dims)

function rand(
    ::Type{T}, dims::Int...
) where {T<:Union{SUPPORTED_FLOAT_TYPES,SUPPORTED_COMPLEX_TYPES,_RNG_INT_TYPES,Bool}}
    return cuNumeric.rand(T, dims)
end
rand(dims::Dims) = cuNumeric.rand(DEFAULT_FLOAT, dims)
rand(dims::Int...) = cuNumeric.rand(DEFAULT_FLOAT, dims)

function _unitrange_bounds(r::AbstractUnitRange{<:Integer})
    lo = Int64(first(r))
    hi = Int64(last(r))
    hi < lo && throw(ArgumentError("empty range $r"))
    return lo, hi + Int64(1) # kernel is [low, high)
end

function rand(r::AbstractUnitRange{T}, dims::Dims) where {T<:_RNG_INT_TYPES}
    lo, hi = _unitrange_bounds(r)
    return integers(get_static_generator(), T, dims; low=lo, high=hi)
end

rand(r::AbstractUnitRange{<:_RNG_INT_TYPES}, dims::Int...) = cuNumeric.rand(r, dims)

function rand(r::AbstractUnitRange{Bool}, dims::Dims)
    first(r) == last(r) && return fill(first(r), dims)
    first(r) == false && last(r) == true && return cuNumeric.rand(Bool, dims)
    return throw(ArgumentError("empty range $r"))
end
rand(r::AbstractUnitRange{Bool}, dims::Int...) = cuNumeric.rand(r, dims)

@doc"""
    Random.randn!(arr::NDArray{<:SUPPORTED_FLOAT_TYPES})
    Random.randn!(arr::NDArray{<:SUPPORTED_COMPLEX_TYPES})

Fill `arr` in-place with standard normal samples. Real arrays have mean 0 and
variance 1. Complex arrays match Julia `randn(Complex{T})`: independent real
and imag parts with variance `1/2`, so `E[|z|²] = 1`.
"""
function Random.randn!(arr::NDArray{<:SUPPORTED_FLOAT_TYPES})
    return randn!(get_static_generator(), arr)
end

function Random.randn!(arr::NDArray{<:SUPPORTED_COMPLEX_TYPES})
    return randn!(get_static_generator(), arr)
end

function Random.randn!(arr::NDArray{T}) where {T}
    return error(
        "randn! only supports Float32, Float64, ComplexF32, and ComplexF64 NDArray storage"
    )
end

@doc"""
    cuNumeric.randn([T=Float32,] dims::Int...)
    cuNumeric.randn([T=Float32,] dims::Tuple)

Create a new `NDArray` filled with standard normal samples. Complex types
match Julia `randn(Complex{T})` (`E[|z|²] = 1`).

Uses a process-global [`XORWOW`](@ref cuNumeric.XORWOW) generator. For a different
engine, pass a [`Generator`](@ref cuNumeric.Generator) to `randn!`. Shift or scale
in user code (`μ .+ σ .* Z`); `loc`/`scale` are not part of the public API.

# Examples
```@repl
cuNumeric.randn(2, 2)
cuNumeric.randn(Float64, 1000)
```
"""
function randn(::Type{T}, dims::Dims) where {T<:SUPPORTED_FLOAT_TYPES}
    return randn(get_static_generator(), T, dims)
end

function randn(::Type{T}, dims::Dims) where {T<:SUPPORTED_COMPLEX_TYPES}
    return randn(get_static_generator(), T, dims)
end

function randn(
    ::Type{T}, dims::Int...
) where {T<:Union{SUPPORTED_FLOAT_TYPES,SUPPORTED_COMPLEX_TYPES}}
    return cuNumeric.randn(T, dims)
end
randn(dims::Dims) = cuNumeric.randn(DEFAULT_FLOAT, dims)
randn(dims::Int...) = cuNumeric.randn(DEFAULT_FLOAT, dims)

@doc"""
    Random.randexp!(arr::NDArray{<:SUPPORTED_FLOAT_TYPES})

Fill `arr` in-place with exponential samples of scale 1 (mean 1), matching
Julia `randexp`.
"""
function Random.randexp!(arr::NDArray{<:SUPPORTED_FLOAT_TYPES})
    return randexp!(get_static_generator(), arr)
end

function Random.randexp!(arr::NDArray{T}) where {T}
    return error("randexp! only supports Float32 and Float64 NDArray storage")
end

@doc"""
    cuNumeric.randexp([T=Float32,] dims::Int...)
    cuNumeric.randexp([T=Float32,] dims::Tuple)

Create a new `NDArray` filled with exponential samples of scale 1 (mean 1),
matching Julia `randexp`.

Uses a process-global [`XORWOW`](@ref cuNumeric.XORWOW) generator. For a
different engine or scale, use `randexp!(generator, arr; scale)`.

# Examples
```@repl
cuNumeric.randexp(2, 2)
cuNumeric.randexp(Float64, 1000)
```
"""
function randexp(::Type{T}, dims::Dims) where {T<:SUPPORTED_FLOAT_TYPES}
    return randexp(get_static_generator(), T, dims)
end

randexp(::Type{T}, dims::Int...) where {T<:SUPPORTED_FLOAT_TYPES} = cuNumeric.randexp(T, dims)
randexp(dims::Dims) = cuNumeric.randexp(DEFAULT_FLOAT, dims)
randexp(dims::Int...) = cuNumeric.randexp(DEFAULT_FLOAT, dims)
