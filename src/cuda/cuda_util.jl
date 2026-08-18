const KERNEL_OFFSET = sizeof(CUDACore.KernelState)
const _COMPATIBLE_PTX_VERSION = Ref{VersionNumber}()

function _select_compatible_ptx_version()
    compiler_version = CUDACore.compiler_version()
    driver_version = CUDACore.driver_version()
    compatible = intersect(
        Set(CUDACore.llvm_compat().ptx),
        Set(CUDACore.ptxas_compat(compiler_version).ptx),
        Set(CUDACore.ptxas_compat(driver_version).ptx),
    )
    filter!(>=(v"6.2"), compatible)
    isempty(compatible) && error(
        "No PTX ISA is supported by the active NVPTX backend, " *
        "CUDA compiler $compiler_version, and CUDA driver $driver_version",
    )
    ptx = maximum(compatible)
    @debug "Selected compatible PTX ISA" ptx compiler_version driver_version
    return ptx
end

function _setup_cuda_tasking()
    if CUDACore.functional()
        _COMPATIBLE_PTX_VERSION[] = _select_compatible_ptx_version()
        # in cuda.jl to notify /wrapper/src/cuda.cpp about CUDA.jl kernel state size
        register_kernel_state_size(UInt64(KERNEL_OFFSET))
        # in /wrapper/src/cuda.cpp
        register_tasks()
    else
        @warn "CUDA.jl is not functional; skipping CUDA kernel registration."
    end
end

# Dense @cuda_task / RunPTXTask — MUST match CUDA.jl CuDeviceArray layout.
# Other memory types: https://github.com/JuliaGPU/CUDA.jl/blob/345c1600ebd561135148bb04ee2657f521a40e25/CUDACore/src/device/pointer.jl#L7
function ndarray_cuda_type(::Type{<:NDArray{T,N}}) where {T,N}
    return CUDACore.CuDeviceArray{T,N,CUDACore.AS.Global}
end

function ndarray_cuda_type(::Type{T}) where {T}
    Base.isbitstype(T) || throw(ArgumentError("Unsupported argument type: $(T)"))
    return T
end

"""
    map_cuda_type(::Type{T})::Type

Recursively rewrite cuNumeric broadcast-related types for fused-broadcast PTX
(e.g. mapping `NDArray{...}` to `CuStridedDeviceArray{...}`). Dense `@cuda_task`
uses `ndarray_cuda_type` → CUDA.jl `CuDeviceArray` instead.
"""
map_cuda_type(::Type{T}) where {T} = T

map_cuda_type(::Type{<:NDArray{T,N}}) where {T,N} = CuStridedDeviceArray{T,N,CUDACore.AS.Global}

function map_cuda_type(::Type{T}) where {T<:Tuple}
    return Tuple{map_cuda_type.(T.parameters)...}
end

function map_cuda_type(::Type{Base.Broadcast.Broadcasted{S,Ax,F,Args}}) where {S,Ax,F,Args}
    return Base.Broadcast.Broadcasted{map_cuda_type(S),Ax,F,map_cuda_type(Args)}
end

function map_cuda_type(::Type{Base.Broadcast.Extruded{X,K,D}}) where {X,K,D}
    return Base.Broadcast.Extruded{map_cuda_type(X),K,D}
end
