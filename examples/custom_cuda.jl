using cuNumeric

using CUDA
import CUDA: i32

function kernel_add(a, b, c, N)
    i = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    if i <= N
        @inbounds c[i] = a[i] + b[i]
    end
    return nothing
end

N = 1024
threads = 256
blocks = cld(N, threads)

a = cuNumeric.full(N, 1.0f0)
b = cuNumeric.full(N, 2.0f0)
c = cuNumeric.ones(Float32, N)

task = cuNumeric.@cuda_task kernel_add(a, b, c, UInt32(1))

cuNumeric.@launch task=task threads=threads blocks=blocks inputs=(a, b) outputs=c scalars=UInt32(N)

c_cpu = c[:]
println("Result of c after kenel launch: ", c_cpu[1])
