using cuNumeric
using CUDA
import CUDA: i32
using Test

function cuNumeric_unfused(u, v, f, k)
    F_u = (
        (
            -u[2:(end - 1), 2:(end - 1)] .*
            (v[2:(end - 1), 2:(end - 1)] .* v[2:(end - 1), 2:(end - 1)])
        ) + f*(1.0f0 .- u[2:(end - 1), 2:(end - 1)])
    )

    F_v = (
        (
            u[2:(end - 1), 2:(end - 1)] .*
            (v[2:(end - 1), 2:(end - 1)] .* v[2:(end - 1), 2:(end - 1)])
        ) - (f+k)*v[2:(end - 1), 2:(end - 1)]
    )

    return F_u, F_v
end

function fused_kernel(u, v, F_u, F_v, N::UInt32, f::Float32, k::Float32)
    i = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1i32) * blockDim().y + threadIdx().y

    if i <= N - 1 && j <= N - 1 # index from 2 --> end - 1
        @inbounds begin
            u_ij = u[i + 1, j + 1]
            v_ij = v[i + 1, j + 1]
            v_sq = v_ij * v_ij
            F_u[i, j] = -u_ij + v_sq + f*(1.0f0 - u_ij)
            F_v[i, j] = u_ij + v_sq - (f + k)*v_ij
        end
    end

    return nothing
end

function run_fused_cunumeric(N, u, v)
    threads2d = (16, 16)  # 16*16 = 256 threads per block
    blocks = (cld(N, threads2d[1]), cld(N, threads2d[2]))

    F_u = cuNumeric.zeros(Float32, (N-2, N-2))
    F_v = cuNumeric.zeros(Float32, (N-2, N-2))

    f = 0.03f0
    k = 0.06f0

    task = cuNumeric.@cuda_task fused_kernel(u, v, F_u, F_v, UInt32(N), f, k)

    cuNumeric.@launch task=task threads=threads2d blocks=blocks inputs=(u, v) outputs=(F_u, F_v) scalars=(
        UInt32(N), f, k
    )

    return F_u, F_v
end

function run_baseline(N, u, v)
    threads2d = (16, 16)  # 16*16 = 256 threads per block
    blocks = (cld(N, threads2d[1]), cld(N, threads2d[2]))

    F_u = CUDA.zeros(Float32, (N-2, N-2))
    F_v = CUDA.zeros(Float32, (N-2, N-2))

    f = 0.03f0
    k = 0.06f0

    @cuda threads=threads2d blocks=blocks fused_kernel(u, v, F_u, F_v, UInt32(N), f, k)

    return F_u, F_v
end

function run_unfused(N, u, v)
    f = 0.03f0
    k = 0.06f0

    F_u, F_v = cuNumeric_unfused(u, v, f, k)

    return F_u, F_v
end

function fusion_test(; N=32, atol=1.0f-6, rtol=1.0f-6)
    u = cuNumeric.random(Float32, (N, N))
    v = cuNumeric.random(Float32, (N, N))

    # u_base = CUDA.rand(Float32, (N, N))
    # v_base = CUDA.rand(Float32, (N, N))
    # base_u, base_v = run_baseline(N, u_base, v_base)

    fused_u, fused_v = run_fused_cunumeric(N, u, v)
    unfused_u, unfused_v = run_unfused(N, u, v)

    @assert fused_u == unfused_u
    @assert fused_v == unfused_v

    # trying to debug why the above fails
    cpu_fused_u = fused_u[:, :]
    cpu_fused_v = fused_v[:, :]

    cpu_unfused_u = unfused_u[:, :]
    cpu_unfused_v = unfused_v[:, :]

    @test isapprox(cpu_fused_u, cpu_unfused_u; atol=atol, rtol=rtol)
    @test isapprox(cpu_fused_v, cpu_unfused_v; atol=atol, rtol=rtol)
end

fusion_test()
