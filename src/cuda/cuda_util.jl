const KERNEL_OFFSET = sizeof(CUDACore.KernelState)

function _setup_cuda_tasking()
    if CUDACore.functional()
        # in cuda.jl to notify /wrapper/src/cuda.cpp about CUDA.jl kernel state size
        register_kernel_state_size(UInt64(KERNEL_OFFSET))
        # in /wrapper/src/cuda.cpp
        register_tasks()
    else
        @warn "CUDA.jl is not functional; skipping CUDA kernel registration."
    end
end

# Other memeory types here: https://github.com/JuliaGPU/CUDA.jl/blob/345c1600ebd561135148bb04ee2657f521a40e25/CUDACore/src/device/pointer.jl#L7
function ndarray_cuda_type(::Type{<:NDArray{T,N}}) where {T,N}
    CUDACore.CuDeviceArray{T,N,CUDACore.AS.Global}
end

function ndarray_cuda_type(::Type{T}) where {T}
    Base.isbitstype(T) || throw(ArgumentError("Unsupported argument type: $(T)"))
    return T
end

"""
    map_cuda_type(::Type{T})::Type

Recursively rewrite cuNumeric broadcast-related types so they can be treated as CUDA-friendly
types for code generation (e.g. mapping `NDArray{...}` to `CuDeviceArray{...}` inside
`Base.Broadcast.Broadcasted{...}` type parameters).
"""
map_cuda_type(::Type{T}) where {T} = T

map_cuda_type(::Type{<:NDArray{T,N}}) where {T,N} = ndarray_cuda_type(NDArray{T,N})

function map_cuda_type(::Type{T}) where {T<:Tuple}
    return Tuple{map_cuda_type.(T.parameters)...}
end

function map_cuda_type(::Type{Base.Broadcast.Broadcasted{S,Ax,F,Args}}) where {S,Ax,F,Args}
    return Base.Broadcast.Broadcasted{map_cuda_type(S),Ax,F,map_cuda_type(Args)}
end
