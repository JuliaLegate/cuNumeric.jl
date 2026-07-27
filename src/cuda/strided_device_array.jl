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
=#

# Device-side strided array packed by RunPTXBroadcastTask only.
# Layout must match C++ `CuStridedDeviceArray<D>` in
# lib/cunumeric_jl_wrapper/src/cuda.cpp:
#   ptr, maxsize, dims[N], strides[N] (element strides), length
#
# Dense RunPTXTask / @cuda_task still uses CUDA.jl CuDeviceArray (unchanged).

struct CuStridedDeviceArray{T,N,A} <: AbstractArray{T,N}
    ptr::CUDACore.LLVMPtr{T,A}
    maxsize::Int
    dims::Dims{N}
    strides::Dims{N}
    len::Int
end

Base.elsize(::Type{<:CuStridedDeviceArray{T}}) where {T} = sizeof(T)
Base.size(a::CuStridedDeviceArray) = a.dims
Base.size(a::CuStridedDeviceArray{<:Any,1}) = (a.len,)
Base.length(a::CuStridedDeviceArray) = a.len
Base.IndexStyle(::Type{<:CuStridedDeviceArray}) = IndexLinear()

function Base.pointer(a::CuStridedDeviceArray{T,<:Any,A}) where {T,A}
    return Base.unsafe_convert(CUDACore.LLVMPtr{T,A}, a)
end
function Base.unsafe_convert(
    ::Type{CUDACore.LLVMPtr{T,A}}, a::CuStridedDeviceArray{T,<:Any,A}
) where {T,A}
    return a.ptr
end

# 0-based element offset from a 1-based linear index in Julia column-major order
# over `dims`, using Legate element `strides`.
#
# Must not use checked rem/div or signed↔unsigned converts — those emit
# DivideError / InexactError → gpu_report_exception and break LoadPTX.
@inline _bitcast_uint(x::Int) = reinterpret(UInt, x)
@inline _bitcast_int(x::UInt) = reinterpret(Int, x)

@inline function _strided_elem_offset(dims::Dims{N}, strides::Dims{N}, I::Integer) where {N}
    idx = _bitcast_uint(Int(I) - 1)
    off = 0
    @inbounds for d in 1:N
        dlen = _bitcast_uint(Int(dims[d]))
        c = _bitcast_int(Core.Intrinsics.urem_int(idx, dlen))
        idx = Core.Intrinsics.udiv_int(idx, dlen)
        off += c * Int(strides[d])
    end
    return off
end

@inline function _strided_elem_offset(strides::Dims{N}, I::CartesianIndex{N}) where {N}
    off = 0
    @inbounds for d in 1:N
        off += (Int(I[d]) - 1) * Int(strides[d])
    end
    return off
end

@inline _strided_align(::CuStridedDeviceArray{T}) where {T} = Base.datatype_alignment(T)

# No @boundscheck / throw — those emit gpu_report_exception and break LoadPTX.
CUDACore.@device_function @inline function _strided_arrayref(
    A::CuStridedDeviceArray{T}, index::Integer
) where {T}
    off = _strided_elem_offset(A.dims, A.strides, index)
    return unsafe_load(pointer(A), off + 1, Val(_strided_align(A)))
end

CUDACore.@device_function @inline function _strided_arrayset(
    A::CuStridedDeviceArray{T}, x::T, index::Integer
) where {T}
    off = _strided_elem_offset(A.dims, A.strides, index)
    unsafe_store!(pointer(A), x, off + 1, Val(_strided_align(A)))
    return A
end

Base.@propagate_inbounds Base.getindex(A::CuStridedDeviceArray{T}, i::Integer) where {T} =
    _strided_arrayref(
        A, i
    )
Base.@propagate_inbounds function Base.setindex!(
    A::CuStridedDeviceArray{T}, x, i::Integer
) where {T}
    return _strided_arrayset(A, convert(T, x)::T, i)
end

Base.to_index(::CuStridedDeviceArray, i::Integer) = i

Base.@propagate_inbounds function Base.getindex(
    A::CuStridedDeviceArray{T,N}, I::CartesianIndex{N}
) where {T,N}
    off = _strided_elem_offset(A.strides, I)
    return unsafe_load(pointer(A), off + 1, Val(_strided_align(A)))
end

Base.@propagate_inbounds function Base.setindex!(
    A::CuStridedDeviceArray{T,N}, x, I::CartesianIndex{N}
) where {T,N}
    off = _strided_elem_offset(A.strides, I)
    unsafe_store!(pointer(A), convert(T, x)::T, off + 1, Val(_strided_align(A)))
    return A
end
