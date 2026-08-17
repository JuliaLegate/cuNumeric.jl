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

# cuFFT mixed-radix kernels only cover lengths whose prime factors are all
# <= 131. A larger factor forces Bluestein (slower, more scratch).
const CUFFT_MAX_EFFICIENT_PRIME = 131
const _CUFFT_SMALL_PRIMES = (
    2,
    3,
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
    53,
    59,
    61,
    67,
    71,
    73,
    79,
    83,
    89,
    97,
    101,
    103,
    107,
    109,
    113,
    127,
    131,
)

function _has_large_prime_factor(n::Integer)
    n < 2 && return false
    m = Int(n)
    for p in _CUFFT_SMALL_PRIMES
        while m % p == 0
            m ÷= p
        end
        m == 1 && return false
    end
    return true
end

function _unique_axes(axes0::NTuple{R,Int64}) where {R}
    seen = zero(UInt64)
    out = Vector{Int64}(undef, R)
    n = 0
    for ax in axes0
        bit = one(UInt64) << ax
        if seen & bit == 0
            seen |= bit
            n += 1
            out[n] = ax
        end
    end
    return resize!(out, n)
end

function _axes_sorted(axes0::NTuple{R,Int64}) where {R}
    for i in 2:R
        axes0[i] < axes0[i - 1] && return false
    end
    return true
end

function _operate_over_axes(axes0::NTuple{R,Int64}, ndim::Int) where {R}
    unique_axes = _unique_axes(axes0)
    return length(unique_axes) != R || R != ndim || !_axes_sorted(axes0)
end

function _bluestein_mask(
    axes0::NTuple{R,Int64}, in_size::NTuple{N,Int}, out_size::NTuple{N,Int}
) where {R,N}
    mask = Int32(0)
    slow = Tuple{Int,Int}[]
    seen = zero(UInt64)
    for ax in axes0
        bit = one(UInt64) << ax
        seen & bit != 0 && continue
        seen |= bit
        jdim = Int(ax) + 1
        length_ax = max(in_size[jdim], out_size[jdim])
        if _has_large_prime_factor(length_ax)
            mask |= Int32(1) << ax
            push!(slow, (ax, length_ax))
        end
    end
    if !isempty(slow)
        details = join(("axis $ax (length $len)" for (ax, len) in slow), ", ")
        @warn "cuNumeric is computing an FFT over $details whose length has a prime factor > $CUFFT_MAX_EFFICIENT_PRIME, so cuFFT falls back to the Bluestein algorithm. You may notice significantly decreased performance and much higher GPU memory usage. Zero-padding the transformed axis to a length whose prime factors are all <= $CUFFT_MAX_EFFICIENT_PRIME (e.g. the next power of two) avoids this."
    end
    return mask
end

function _assert_fft_gpu()
    Legate.num_gpus() > 0 && return nothing
    return throw(
        ErrorException(
            "FFT requires a CUDA GPU; cupynumeric's FFT task has no CPU variant"
        ),
    )
end

_fft_kind(::Type{ComplexF32}) = Int32(cuNumeric.FFT_C2C)
_fft_kind(::Type{ComplexF64}) = Int32(cuNumeric.FFT_Z2Z)

function _fft_dims(::NDArray{<:Any,N}) where {N}
    N >= 1 || throw(ArgumentError("fft does not support 0-dimensional arrays"))
    return ntuple(identity, Val(N))
end

function _fft_dims(A::NDArray, dim::Integer)
    return _fft_dims(A, (Int(dim),))
end

function _fft_dims(A::NDArray{T,N}, dims) where {T,N}
    N >= 1 || throw(ArgumentError("fft does not support 0-dimensional arrays"))
    R = length(dims)
    R >= 1 || throw(ArgumentError("fft dims must contain at least one dimension"))
    region = ntuple(i -> Int(dims[i]), R)
    seen = zero(UInt64)
    for d in region
        (1 <= d <= N) || throw(ArgumentError("fft dim $d is out of range for a $N-d array"))
        bit = one(UInt64) << d
        seen & bit != 0 && throw(ArgumentError("fft dims must be unique; got $region"))
        seen |= bit
    end
    return region
end

# Leading dimension is the batch; every remaining dim is transformed.
# Same CUPYNUMERIC_FFT task as `fft` — the batch axis is the one that may split.
function _fft_batch_dims(::NDArray{<:Any,N}) where {N}
    N >= 2 || throw(
        ArgumentError(
            "batched_fft requires a leading batch dimension; got a $N-d array. Use fft for a single transform."
        ),
    )
    return ntuple(i -> i + 1, Val(N - 1))
end

function _ifft_scale(::Type{T}, sz::NTuple{N,Int}, dims::NTuple{R,Int}) where {T,N,R}
    n = one(real(T))
    for d in dims
        n *= real(T)(sz[d])
    end
    return T(inv(n))
end

_fft_scope_name(direction::Int32) =
    direction == Int32(cuNumeric.FFT_INVERSE) ? "ifft" : "fft"

"""
    fft_task!(out, inp, dims, direction; scale=false)

Launch cupynumeric's `CUPYNUMERIC_FFT` auto task. `dims` are 1-based Julia
dimensions. `out` and `inp` must have the same shape and complex eltype; they
may be the same array (in-place C2C). When `scale` is true the inverse is
normalized in this same task scope (`out .*= 1/N`).
"""
function fft_task!(
    out::NDArray{T,N},
    inp::NDArray{T,N},
    dims::NTuple{R,Int},
    direction::Int32;
    scale::Bool=false,
) where {T<:SUPPORTED_COMPLEX_TYPES,N,R}
    _assert_fft_gpu()
    size(out) == size(inp) ||
        throw(DimensionMismatch("FFT output size $(size(out)) != input size $(size(inp))"))

    axes0 = ntuple(i -> Int64(dims[i] - 1), Val(R))
    unique_axes = _unique_axes(axes0)
    operate_over = _operate_over_axes(axes0, N)
    # Warn on awkward lengths. cuFFT may take Bluestein internally; the
    # 26.06 task has no bluestein_mask scalar, so we do not send one.
    _bluestein_mask(axes0, size(inp), size(out))
    kind = _fft_kind(T)

    @task_scope _fft_scope_name(direction) begin
        rt = Legate.get_runtime()
        lib = cuNumeric.get_lib()
        task = Legate.create_auto_task(rt, lib, cuNumeric.FFT)
        cuNumeric.task_throws_exception(task, true)

        l_out = nda_to_logical_array(out)
        l_in = inp === out ? l_out : nda_to_logical_array(inp)

        out_var = Legate.add_output(task, l_out)
        in_var = Legate.add_input(task, l_in)

        # 26.06 fft_template.inl: kind, direction, operate_over_axes, then axes.
        Legate.add_scalar(task, Legate.Scalar(kind))
        Legate.add_scalar(task, Legate.Scalar(direction))
        Legate.add_scalar(task, Legate.Scalar(operate_over))
        for ax in axes0
            Legate.add_scalar(task, Legate.Scalar(ax))
        end

        Legate.add_constraint(task, Legate.align(out_var, in_var))
        if N > length(unique_axes)
            Legate.add_broadcast(task, l_in, CxxWrap.StdVector(UInt32.(unique_axes)))
        else
            Legate.add_broadcast(task, l_in)
        end

        Legate.submit_auto_task(rt, task)
        if scale
            out .*= _ifft_scale(T, size(out), dims)
        end
    end
    return out
end
