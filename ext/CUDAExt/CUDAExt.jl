module CUDAExt

using Random
using CUDA
import Legate
import cuNumeric
import cuNumeric: @cuda_task, @launch, NDArray


const KERNEL_OFFSET = sizeof(CUDA.KernelState)

include("cuda.jl")

function __init__()
    if CUDA.functional()
        # in cuda.jl to notify /wrapper/src/cuda.cpp about CUDA.jl kernel state size
        cuNumeric.register_kernel_state_size(UInt64(KERNEL_OFFSET))
        # in /wrapper/src/cuda.cpp
        cuNumeric.register_tasks();
    else
        @warn "CUDA.jl is not functional; skipping CUDA kernel registration."
    end
end

end # module CUDAExt