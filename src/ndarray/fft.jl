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
=#

export fft, ifft, fft!, ifft!, batched_fft, batched_ifft, batched_fft!, batched_ifft!

const _FFT_PROMOTABLE = Union{SUPPORTED_INT_TYPES,Bool,SUPPORTED_FLOAT_TYPES}
const _FFT_ACCEPTED = Union{SUPPORTED_COMPLEX_TYPES,_FFT_PROMOTABLE}

_fft_eltype(::Type{ComplexF32}) = ComplexF32
_fft_eltype(::Type{ComplexF64}) = ComplexF64
_fft_eltype(::Type{Float32}) = ComplexF32
_fft_eltype(::Type{Float64}) = ComplexF64
_fft_eltype(::Type{T}) where {T<:_FFT_PROMOTABLE} = ComplexF64

function _fft_complex(A::NDArray{T}) where {T<:_FFT_ACCEPTED}
    return unchecked_promote_arr(A, _fft_eltype(T))
end

function _fft_out_of_place(
    A::NDArray{<:_FFT_ACCEPTED}, dims::NTuple{R,Int}, direction::Int32, scale::Bool
) where {R}
    inp = _fft_complex(A)
    out = cuNumeric.zeros(eltype(inp), size(inp))
    fft_task!(out, inp, dims, direction; scale)
    # Promotion allocated a new complex buffer; the task already holds the store.
    inp !== A && destroy!(inp)
    return out
end

"""
    fft(A::NDArray, [dims])

Unnormalized complex discrete Fourier transform of `A`, matching
`AbstractFFTs.fft`. By default every dimension is transformed. `dims` selects a
subset (Julia 1-based dimensions).

Real and integer inputs are converted to complex (`Float32` → `ComplexF32`,
everything else → `ComplexF64`). The result has the same shape as `A`.

This is GPU-only: cupynumeric's FFT task has no CPU variant. Multi-GPU
execution is limited to batching over dimensions that are not transformed.

There is no reusable cuFFT plan; each call launches the `CUPYNUMERIC_FFT` task.
"""
function fft(A::NDArray{<:_FFT_ACCEPTED})
    return _fft_out_of_place(A, _fft_dims(A), Int32(cuNumeric.FFT_FORWARD), false)
end
function fft(A::NDArray{<:_FFT_ACCEPTED}, dims)
    return _fft_out_of_place(A, _fft_dims(A, dims), Int32(cuNumeric.FFT_FORWARD), false)
end

function fft(A::NDArray)
    return throw(ArgumentError("array type $(eltype(A)) is unsupported in fft"))
end
function fft(A::NDArray, ::Any)
    return throw(ArgumentError("array type $(eltype(A)) is unsupported in fft"))
end

"""
    ifft(A::NDArray, [dims])

Normalized inverse discrete Fourier transform of `A`, matching
`AbstractFFTs.ifft`. Equivalent to the unnormalized inverse scaled by `1/N`,
where `N` is the product of the transformed lengths.

See [`fft`](@ref).
"""
function ifft(A::NDArray{<:_FFT_ACCEPTED})
    return _fft_out_of_place(A, _fft_dims(A), Int32(cuNumeric.FFT_INVERSE), true)
end
function ifft(A::NDArray{<:_FFT_ACCEPTED}, dims)
    return _fft_out_of_place(A, _fft_dims(A, dims), Int32(cuNumeric.FFT_INVERSE), true)
end

function ifft(A::NDArray)
    return throw(ArgumentError("array type $(eltype(A)) is unsupported in ifft"))
end
function ifft(A::NDArray, ::Any)
    return throw(ArgumentError("array type $(eltype(A)) is unsupported in ifft"))
end

"""
    fft!(A::NDArray, [dims])

In-place [`fft`](@ref). `A` must already be `ComplexF32` or `ComplexF64`.
"""
function fft!(A::NDArray{T,N}) where {T<:SUPPORTED_COMPLEX_TYPES,N}
    return fft_task!(A, A, _fft_dims(A), Int32(cuNumeric.FFT_FORWARD))
end
function fft!(A::NDArray{T,N}, dims) where {T<:SUPPORTED_COMPLEX_TYPES,N}
    return fft_task!(A, A, _fft_dims(A, dims), Int32(cuNumeric.FFT_FORWARD))
end

function fft!(A::NDArray)
    return throw(ArgumentError("fft! requires a complex NDArray; got $(eltype(A))"))
end
function fft!(A::NDArray, ::Any)
    return throw(ArgumentError("fft! requires a complex NDArray; got $(eltype(A))"))
end

"""
    ifft!(A::NDArray, [dims])

In-place [`ifft`](@ref). `A` must already be `ComplexF32` or `ComplexF64`.
"""
function ifft!(A::NDArray{T,N}) where {T<:SUPPORTED_COMPLEX_TYPES,N}
    return ifft!(A, _fft_dims(A))
end
function ifft!(A::NDArray{T,N}, dims) where {T<:SUPPORTED_COMPLEX_TYPES,N}
    region = _fft_dims(A, dims)
    return fft_task!(A, A, region, Int32(cuNumeric.FFT_INVERSE); scale=true)
end

function ifft!(A::NDArray)
    return throw(ArgumentError("ifft! requires a complex NDArray; got $(eltype(A))"))
end
function ifft!(A::NDArray, ::Any)
    return throw(ArgumentError("ifft! requires a complex NDArray; got $(eltype(A))"))
end

"""
    cuNumeric.batched_fft(A)

FFT every trailing dimension of `A`, treating `size(A, 1)` as a batch.

`A` must be at least 2-d. A `(b, n)` stack is `b` independent 1-d transforms;
`(b, n, m)` is `b` independent 2-d transforms. This is the same
`CUPYNUMERIC_FFT` task as [`fft`](@ref); the leading axis is the one that may
be partitioned across GPUs.
"""
function batched_fft(A::NDArray{<:_FFT_ACCEPTED})
    return fft(A, _fft_batch_dims(A))
end
function batched_fft(A::NDArray)
    return throw(ArgumentError("array type $(eltype(A)) is unsupported in batched_fft"))
end

"""
    cuNumeric.batched_ifft(A)

Normalized inverse of [`batched_fft`](@ref).
"""
function batched_ifft(A::NDArray{<:_FFT_ACCEPTED})
    return ifft(A, _fft_batch_dims(A))
end
function batched_ifft(A::NDArray)
    return throw(ArgumentError("array type $(eltype(A)) is unsupported in batched_ifft"))
end

"""
    cuNumeric.batched_fft!(A)

In-place [`batched_fft`](@ref). `A` must already be complex.
"""
function batched_fft!(A::NDArray{<:SUPPORTED_COMPLEX_TYPES})
    return fft!(A, _fft_batch_dims(A))
end
function batched_fft!(A::NDArray)
    return throw(ArgumentError("batched_fft! requires a complex NDArray; got $(eltype(A))"))
end

"""
    cuNumeric.batched_ifft!(A)

In-place [`batched_ifft`](@ref). `A` must already be complex.
"""
function batched_ifft!(A::NDArray{<:SUPPORTED_COMPLEX_TYPES})
    return ifft!(A, _fft_batch_dims(A))
end
function batched_ifft!(A::NDArray)
    return throw(ArgumentError("batched_ifft! requires a complex NDArray; got $(eltype(A))"))
end
