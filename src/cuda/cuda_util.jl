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

ndarray_cuda_type(::Type{<:NDArray{T,N}}) where {T,N} = CuDeviceArray{T,N,1}

function ndarray_cuda_type(::Type{T}) where {T}
    Base.isbitstype(T) || throw(ArgumentError("Unsupported argument type: $(T)"))
    return T
end
