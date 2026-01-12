module CUDAExt

using Random
using CUDA
using cuNumeric

include("cuda.jl")

function __init__()
    if CUDA.functional()
        # in cuda.jl to notify /wrapper/src/cuda.cpp about CUDA.jl kernel state size
        cuNumeric.set_kernel_state_size();
        # in /wrapper/src/cuda.cpp
        cuNumeric.register_tasks();
    else
        @warn "CUDA.jl is not functional; skipping CUDA kernel registration."
    end
end

end # module CUDAExt