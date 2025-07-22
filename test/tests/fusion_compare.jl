using cuNumeric
using CUDA
import CUDA: i32
using Test

function unfused_cunumeric(u, v, f, k)
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

function unfused_cuda(u, v, f::Float32, k::Float32)
    @views F_u = (
        (
            -u[2:(end - 1), 2:(end - 1)] .*
            (v[2:(end - 1), 2:(end - 1)] .* v[2:(end - 1), 2:(end - 1)])
        ) + f*(1.0f0 .- u[2:(end - 1), 2:(end - 1)])
    )

    @views F_v = (
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
    if i <= N-2 && j <= N-2
        @inbounds begin
            u_ij = u[i + 1, j + 1]
            v_ij = v[i + 1, j + 1]
            v_sq = v_ij * v_ij
            # F_u[i, j] = (-u_ij * v_sq) + f*(1.0f0 - u_ij)
            # F_v[i, j] = (u_ij * v_sq) - (f + k)*v_ij
            F_u[i, j] = i
            F_v[i, j] = j
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

function run_fused_baseline(N, u, v)
    threads2d = (16, 16)  # 16*16 = 256 threads per block
    blocks = (cld(N, threads2d[1]), cld(N, threads2d[2]))

    F_u = CUDA.zeros(Float32, (N-2, N-2))
    F_v = CUDA.zeros(Float32, (N-2, N-2))

    f = 0.03f0
    k = 0.06f0

    @cuda threads=threads2d blocks=blocks fused_kernel(u, v, F_u, F_v, UInt32(N), f, k)

    return F_u, F_v
end

function run_unfused_cunumeric(N, u, v)
    f = 0.03f0
    k = 0.06f0

    F_u, F_v = unfused_cunumeric(u, v, f, k)

    return F_u, F_v
end

function run_unfused_baseline(N, u, v)
    f = 0.03f0
    k = 0.06f0

    F_u, F_v = unfused_cuda(u, v, f, k)

    return F_u, F_v
end

function fusion_test(; N=16, atol=1.0f-6, rtol=1.0f-6)
    u = cuNumeric.random(Float32, (N, N))
    v = cuNumeric.random(Float32, (N, N))

    # using CUDA
    u_base = CUDA.rand(Float32, (N, N))
    v_base = CUDA.rand(Float32, (N, N))
    Fu_base_fused, Fv_base_fused = run_fused_baseline(N, u_base, v_base)
    Fu_base_unfused, Fv_base_unfused = run_unfused_baseline(N, u_base, v_base)
    @test isapprox(Fu_base_fused, Fu_base_unfused; atol=atol, rtol=rtol)
    @test isapprox(Fv_base_fused, Fv_base_unfused; atol=atol, rtol=rtol)

    # using cuNumeric
    Fu_fused, Fv_fused = run_fused_cunumeric(N, u, v)
    cuNumeric.gpu_sync()
    Fu_unfused, Fv_unfused = run_unfused_cunumeric(N, u, v)
    cuNumeric.gpu_sync()

    # @assert Fu_fused == Fu_unfused
    # @assert Fv_fused == Fv_unfused

    # trying to debug why the above fails
    cpu_Fu_fused = Fu_fused[:, :]
    cpu_Fv_fused = Fv_fused[:, :]

    print(cpu_Fu_fused)

    cpu_Fu_unfused = Fu_unfused[:, :]
    cpu_Fv_unfused = Fv_unfused[:, :]

    @test isapprox(cpu_Fu_fused, cpu_Fu_unfused; atol=atol, rtol=rtol)
    @test isapprox(cpu_Fv_fused, cpu_Fv_unfused; atol=atol, rtol=rtol)
end

fusion_test()
