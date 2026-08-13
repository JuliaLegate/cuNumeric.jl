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

"""
    Generator(bit_generator)

cuPyNumeric-style RNG wrapping a [`BitGenerator`](@ref). Module-level
[`rand`](@ref), [`randn`](@ref) use a process-global
XORWOW generator; pass a different engine to `default_rng` for Philox or MRG32k3a.
"""
struct Generator{B<:BitGenerator}
    bit_generator::B
end

const _RNG_INT_TYPES = Union{Int16,Int32,Int64}
const _static_generator = Ref{Union{Nothing,Generator{XORWOW}}}(nothing)

"""
    default_rng()
    default_rng(seed)
    default_rng(::Type{<:BitGenerator}, seed=nothing; flags=0)
    default_rng(bit_generator)
    default_rng(generator)

Return a [`Generator`](@ref). The no-argument and integer-seed methods use
[`XORWOW`](@ref). This does **not** replace the process-global generator used
by module-level `rand` / `randn`.

```julia
g = cuNumeric.default_rng(cuNumeric.PHILOX4_32_10, 1234)
cuNumeric.random(g, Float32, (8, 8))
```
"""
function default_rng()
    return Generator(XORWOW())
end

function default_rng(seed::Integer)
    return Generator(XORWOW(seed))
end

function default_rng(
    ::Type{B}, seed::Union{Integer,Nothing}=nothing; flags::Integer=0
) where {B<:BitGenerator}
    return Generator(B(seed; flags))
end

function default_rng(bg::BitGenerator)
    return Generator(bg)
end

function default_rng(g::Generator)
    return g
end

function get_static_generator()
    gen = _static_generator[]
    if gen === nothing
        gen = default_rng()
        _static_generator[] = gen
    end
    return gen::Generator{XORWOW}
end

function random!(g::Generator, arr::NDArray{Float32})
    _bitgenerator_distribution!(
        arr, g.bit_generator, cuNumeric.BITGENDIST_UNIFORM_32,
        _EMPTY_INT64, SVector{2,Float32}(0, 1), _EMPTY_FLOAT64,
    )
    return arr
end

function random!(g::Generator, arr::NDArray{Float64})
    _bitgenerator_distribution!(
        arr, g.bit_generator, cuNumeric.BITGENDIST_UNIFORM_64,
        _EMPTY_INT64, _EMPTY_FLOAT32, SVector{2,Float64}(0, 1),
    )
    return arr
end

function random!(g::Generator, arr::NDArray{T}) where {T}
    return error("random! only supports NDArray{<:AbstractFloat} of Float32 or Float64")
end

function random(g::Generator, ::Type{T}, dims::Dims) where {T<:AbstractFloat}
    arr = zeros(T, dims)
    random!(g, arr)
    return arr
end

function randn!(g::Generator, arr::NDArray{Float32}; loc::Real=0, scale::Real=1)
    _bitgenerator_distribution!(
        arr, g.bit_generator, cuNumeric.BITGENDIST_NORMAL_32,
        _EMPTY_INT64, SVector{2,Float32}(Float32(loc), Float32(scale)), _EMPTY_FLOAT64,
    )
    return arr
end

function randn!(g::Generator, arr::NDArray{Float64}; loc::Real=0, scale::Real=1)
    _bitgenerator_distribution!(
        arr, g.bit_generator, cuNumeric.BITGENDIST_NORMAL_64,
        _EMPTY_INT64, _EMPTY_FLOAT32, SVector{2,Float64}(Float64(loc), Float64(scale)),
    )
    return arr
end

function randn!(g::Generator, arr::NDArray{T}; loc::Real=0, scale::Real=1) where {T}
    return error("randn! only supports NDArray{<:AbstractFloat} of Float32 or Float64")
end

function randn(
    g::Generator, ::Type{T}, dims::Dims; loc::Real=0, scale::Real=1
) where {T<:AbstractFloat}
    arr = zeros(T, dims)
    randn!(g, arr; loc=loc, scale=scale)
    return arr
end

_integer_distribution(::Type{Int16}) = cuNumeric.BITGENDIST_INTEGERS_16
_integer_distribution(::Type{Int32}) = cuNumeric.BITGENDIST_INTEGERS_32
_integer_distribution(::Type{Int64}) = cuNumeric.BITGENDIST_INTEGERS_64

function _integer_distribution(::Type{T}) where {T}
    return throw(ArgumentError("integer random only supports Int16, Int32, and Int64. Got $T."))
end

# Kernel draws [low, high); Julia `a:b` is mapped to low=a, high=b+1.
function integers!(
    g::Generator, arr::NDArray{T}; low::Integer, high::Integer
) where {T<:_RNG_INT_TYPES}
    Int64(high) <= Int64(low) && throw(ArgumentError("low >= high"))
    _bitgenerator_distribution!(
        arr, g.bit_generator, _integer_distribution(T),
        SVector{2,Int64}(Int64(low), Int64(high)), _EMPTY_FLOAT32, _EMPTY_FLOAT64,
    )
    return arr
end

function integers!(g::Generator, arr::NDArray{T}; low::Integer, high::Integer) where {T}
    return throw(ArgumentError("integer random only supports Int16, Int32, and Int64"))
end

function integers(
    g::Generator, ::Type{T}, dims::Dims; low::Integer, high::Integer
) where {T<:_RNG_INT_TYPES}
    arr = zeros(T, dims)
    integers!(g, arr; low=low, high=high)
    return arr
end
