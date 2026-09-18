# NPB permits changing MK without changing the generated sequence. This uses
# the harness-wide MK=8 so every programming model performs the same batching.
# CUDA.jl remains the intentionally single-GPU baseline.

struct CUDANASEPState{A}
    partials::A
end

function initialize(b::NASEmbarrassinglyParallel{Float64}; mod=CUDA)
    p = validate_nas_ep(b)
    return (CUDANASEPState(CUDA.CuArray{NASEPPartial}(undef, nas_ep_batches(p))),)
end

function cuda_nas_ep_kernel!(partials, jump)
    i = (CUDA.blockIdx().x - 1)*CUDA.blockDim().x + CUDA.threadIdx().x
    i <= length(partials) && (@inbounds partials[i] = nas_ep_batch(i - 1, jump))
    return nothing
end

function run!(b::NASEmbarrassinglyParallel, s::CUDANASEPState)
    threads = 256
    CUDA.@cuda threads=threads blocks=cld(length(s.partials), threads) cuda_nas_ep_kernel!(
        s.partials, nas_ep_batch_jump()
    )
    return s.partials
end

function check_benchmark_correctness(
    b::NASEmbarrassinglyParallel, gs::GlobalSettings; mod=CUDA
)
    state = only(initialize(b; mod))
    result = nas_ep_combine(Array(run!(b, state)))
    return nas_ep_verified(b.class, result.sx, result.sy) ? "pass" : "fail"
end
