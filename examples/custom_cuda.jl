using cuNumeric

using CUDA
import CUDA: i32

cuNumeric.Experimental(true)

function kernel_add(a, b, c, N)
    i = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    if i <= N
        @inbounds c[i] = a[i] + b[i]
    end
    return nothing
end

function kernel_sin(a, b, N)
    i = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    if i <= N
        @inbounds b[i] = sin(a[i])
    end
    return nothing
end

function run_custom_cuda(N=1024)
    threads = 256
    blocks = cld(N, threads)
    a = cuNumeric.fill(1.0f0, N)
    b = cuNumeric.fill(2.0f0, N)
    c = cuNumeric.zeros(Float32, N)
    n_scalar = UInt32(N)

    add_task = cuNumeric.@cuda_task kernel_add(a, b, c, n_scalar)
    cuNumeric.@launch task=add_task threads=threads blocks=blocks inputs=(a, b) outputs=c scalars=n_scalar

    sin_task = cuNumeric.@cuda_task kernel_sin(c, b, n_scalar)
    cuNumeric.@launch task=sin_task threads=threads blocks=blocks inputs=c outputs=b scalars=n_scalar

    allowscalar() do
        return println("sin(1 + 2) = ", b[1])
    end
    return b
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_custom_cuda()
end
