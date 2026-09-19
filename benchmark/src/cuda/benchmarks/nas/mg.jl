# LIMITATION: This idiomatic CUDA.jl baseline expresses the NPB MG operators as
# fused CuArray broadcasts instead of copying NPB-GPU's hand-written kernels.
# CUDA.jl remains the harness's intentionally single-GPU baseline.
# Restriction/interpolation use strided views and multiple broadcasts, not
# JACC's one-kernel-per-operator approach. The common harness times initial
# zeroing and L2 sum-of-squares, but omits NPB's Linf norm (see nas/README.md).

mutable struct CUDANASMGState{U,R,V}
    u::U
    r::R
    rhs::V
end

function initialize(b::NASMultiGrid{Float64}; mod=CUDA)
    p = validate_nas_mg(b)
    sizes = nas_mg_level_sizes(p)
    u = [CUDA.zeros(Float64, n, n, n) for n in sizes]
    r = [CUDA.zeros(Float64, n, n, n) for n in sizes]
    rhs = CUDA.CuArray(nas_mg_rhs(p))
    return (CUDANASMGState(u, r, rhs),)
end

function cuda_nas_mg_norm2(residual)
    n = size(residual, 1)
    interior = @view residual[2:(n - 1), 2:(n - 1), 2:(n - 1)]
    return sum(abs2, interior; dims=(1, 2, 3))
end

function run!(b::NASMultiGrid, s::CUDANASMGState)
    p = nas_mg_parameters(b.class)
    c = nas_mg_smoother(b.class)
    foreach(x -> fill!(x, 0.0), s.u)
    nas_mg_resid!(s.r[end], s.u[end], s.rhs)
    cuda_nas_mg_norm2(s.r[end])
    for _ in 1:p.niter
        nas_mg_cycle!(s.u, s.r, s.rhs, c)
        nas_mg_resid!(s.r[end], s.u[end], s.rhs)
    end
    return cuda_nas_mg_norm2(s.r[end])
end

function check_benchmark_correctness(b::NASMultiGrid, gs::GlobalSettings; mod=CUDA)
    state = only(initialize(b; mod))
    squared = run!(b, state)
    norm = sqrt(only(Array(squared)) / Float64(b.N)^3)
    return nas_mg_verified(b.class, norm) ? "pass" : "fail"
end
