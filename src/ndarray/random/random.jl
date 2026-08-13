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
    Random.rand!(arr::NDArray{<:AbstractFloat})
    Random.rand!(arr::NDArray{<:Union{Int16,Int32,Int64}})

Fill `arr` in-place with uniform random values.

Floating arrays are filled from `[0, 1)`. Integer arrays use the full range of
the element type, matching Julia `rand(T)`.
"""
function Random.rand!(arr::NDArray{<:AbstractFloat})
    return random!(get_static_generator(), arr)
end

function Random.rand!(arr::NDArray{T}) where {T<:_RNG_INT_TYPES}
    return integers!(get_static_generator(), arr; low=typemin(T), high=typemax(T))
end

function Random.rand!(arr::NDArray{T}) where {T}
    return error("rand! supports Float32, Float64, Int16, Int32, and Int64 NDArray storage")
end

@doc"""
    cuNumeric.rand([T=Float32,] dims::Int...)
    cuNumeric.rand([T=Float32,] dims::Tuple)
    cuNumeric.rand(r::AbstractUnitRange, dims...)

Create a new `NDArray` filled with uniform random values.

Floating types (`Float32` / `Float64`) are drawn natively in `[0, 1)`.
Integer types `Int16`, `Int32`, and `Int64` use the full range of `T`.
A unit range (`1:10`) draws inclusive integers, matching Julia `rand(1:10, dims...)`.

Uses a process-global [`XORWOW`](@ref) generator. For a different engine or
seed, see [`default_rng`](@ref) and [`Generator`](@ref).

# Examples
```@repl
cuNumeric.rand(2, 2)
cuNumeric.rand((4, 1))
cuNumeric.rand(0:9, 4, 4)
A = cuNumeric.zeros(Float32, 2, 2); cuNumeric.rand!(A)
```
"""
function rand(::Type{T}, dims::Dims) where {T<:AbstractFloat}
    return random(get_static_generator(), T, dims)
end

function rand(::Type{T}, dims::Dims) where {T<:_RNG_INT_TYPES}
    return integers(get_static_generator(), T, dims; low=typemin(T), high=typemax(T))
end

function rand(::Type{T}, dims::Int...) where {T<:Union{AbstractFloat,_RNG_INT_TYPES}}
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

@doc"""
    Random.randn!(arr::NDArray{<:AbstractFloat})

Fill `arr` in-place with standard normal samples (mean 0, variance 1).
"""
function Random.randn!(arr::NDArray{<:AbstractFloat})
    return randn!(get_static_generator(), arr)
end

function Random.randn!(arr::NDArray{T}) where {T}
    return error("randn! only supports NDArray{<:AbstractFloat} of Float32 or Float64")
end

@doc"""
    cuNumeric.randn([T=Float32,] dims::Int...)
    cuNumeric.randn([T=Float32,] dims::Tuple)

Create a new `NDArray` filled with standard normal samples.

Uses a process-global [`XORWOW`](@ref) generator. For a different engine, loc,
or scale, use `randn!(generator, arr; loc, scale)`.

# Examples
```@repl
cuNumeric.randn(2, 2)
cuNumeric.randn(Float64, 1000)
```
"""
function randn(::Type{T}, dims::Dims) where {T<:AbstractFloat}
    return randn(get_static_generator(), T, dims)
end

randn(::Type{T}, dims::Int...) where {T<:AbstractFloat} = cuNumeric.randn(T, dims)
randn(dims::Dims) = cuNumeric.randn(DEFAULT_FLOAT, dims)
randn(dims::Int...) = cuNumeric.randn(DEFAULT_FLOAT, dims)
