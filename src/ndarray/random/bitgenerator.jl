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

# Mirrors cupynumeric.random._bitgenerator.BitGenerator and the three
# engines Python actually subclasses: XORWOW, MRG32k3a, PHILOX4_32_10.
# CREATE is lazy: the first DISTRIBUTION task materializes the generator.

"""
    BitGenerator

Abstract supertype for cuRAND engines used by [`Generator`](@ref cuNumeric.Generator). Concrete
types are [`XORWOW`](@ref cuNumeric.XORWOW) (default), [`MRG32k3a`](@ref cuNumeric.MRG32k3a), and
[`PHILOX4_32_10`](@ref cuNumeric.PHILOX4_32_10).
"""
abstract type BitGenerator end

const _bitgen_id_lock = ReentrantLock()
const _next_bitgen_id = Ref{Int32}(0)
const _bitgen_zombies = Int32[]

function _next_bitgenerator_id()
    return lock(_bitgen_id_lock) do
        _next_bitgen_id[] += Int32(1)
        return _next_bitgen_id[]
    end
end

function _record_bitgenerator_zombie(handle::Int32)
    handle == 0 && return nothing
    return lock(_bitgen_id_lock) do
        push!(_bitgen_zombies, handle)
        return nothing
    end
end

# C-order strides (NDArray stores are row-major). Julia col-major is only
# handled in Array conversions. DISTRIBUTION writes a dense ptr and ignores this.
function _c_order_strides(shape::NTuple{N,Int}) where {N}
    return SVector{N,Int64}(
        ntuple(Val(N)) do i
            p = Int64(1)
            @inbounds for d in (i + 1):N
                p *= Int64(shape[d])
            end
            return p
        end,
    )
end

_c_order_strides(::Tuple{}) = SVector{0,Int64}()

const _EMPTY_INT64 = SVector{0,Int64}()
const _EMPTY_FLOAT32 = SVector{0,Float32}()
const _EMPTY_FLOAT64 = SVector{0,Float64}()

function _make_bitgenerator(::Type{B}, seed, flags) where {B<:BitGenerator}
    handle = _next_bitgenerator_id()
    seed64 = seed === nothing ? UInt64(time_ns()) : UInt64(seed)
    bg = B(handle, seed64, UInt32(flags))
    finalizer(_finalize_bitgenerator!, bg)
    return bg
end

function _finalize_bitgenerator!(bg::BitGenerator)
    handle = bg.handle
    bg.handle = Int32(0)
    _record_bitgenerator_zombie(handle)
    return nothing
end

"""
    XORWOW(seed=nothing; flags=0)

Default cuRAND BitGenerator (xorwow). `seed=nothing` draws from `time_ns()`.
"""
mutable struct XORWOW <: BitGenerator
    handle::Int32
    seed::UInt64
    flags::UInt32
end
function XORWOW(seed::Union{Integer,Nothing}=nothing; flags::Integer=0)
    return _make_bitgenerator(XORWOW, seed, flags)
end

"""
    MRG32k3a(seed=nothing; flags=0)

cuRAND MRG32k3a BitGenerator.
"""
mutable struct MRG32k3a <: BitGenerator
    handle::Int32
    seed::UInt64
    flags::UInt32
end
function MRG32k3a(seed::Union{Integer,Nothing}=nothing; flags::Integer=0)
    return _make_bitgenerator(MRG32k3a, seed, flags)
end

"""
    PHILOX4_32_10(seed=nothing; flags=0)

cuRAND Philox4_32_10 BitGenerator.
"""
mutable struct PHILOX4_32_10 <: BitGenerator
    handle::Int32
    seed::UInt64
    flags::UInt32
end
function PHILOX4_32_10(seed::Union{Integer,Nothing}=nothing; flags::Integer=0)
    return _make_bitgenerator(PHILOX4_32_10, seed, flags)
end

generator_type(::XORWOW) = cuNumeric.BITGENTYPE_XORWOW
generator_type(::MRG32k3a) = cuNumeric.BITGENTYPE_MRG32K3A
generator_type(::PHILOX4_32_10) = cuNumeric.BITGENTYPE_PHILOX4_32_10

function _bitgenerator_distribution!(
    arr::NDArray,
    bg::BitGenerator,
    distribution,
    intparams::SVector{NI,Int64},
    floatparams::SVector{NF,Float32},
    doubleparams::SVector{ND,Float64},
) where {NI,NF,ND}
    return nda_bitgenerator_distribution!(
        arr,
        bg.handle,
        UInt32(generator_type(bg)),
        bg.seed,
        bg.flags,
        UInt32(distribution),
        _c_order_strides(size(arr)),
        intparams,
        floatparams,
        doubleparams,
    )
end
