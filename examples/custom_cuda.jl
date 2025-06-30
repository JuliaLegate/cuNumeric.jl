using cuNumeric
using Legate
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

a_cpu = a[:]
println("a: ", a_cpu[1])
b_cpu = b[:]
println("b: ", b_cpu[1])
c_cpu = c[:]
println("c: ", c_cpu[1])

task = cuNumeric.@cuda_task kernel_add(a, b, c, UInt32(1))

in = cuNumeric.VectorNDArray()
cuNumeric.push_back(in, a)
cuNumeric.push_back(in, b)

out = cuNumeric.VectorNDArray()
cuNumeric.push_back(out, c)

scalars = Legate.VectorScalar()
Legate.push_back(scalars, Legate.Scalar(UInt32(N)))

cuNumeric.new_task(task.func, in, out, scalars)

# c = cuNumeric.new_task(task.func, (a, b), c, UInt32(N))

cuNumeric.gpu_sync()
# cuNumeric.@launch task=task threads=threads blocks=blocks kernel_add(a, b, c, Int32(1))
c_cpu = c[:]
println("Result of c after kenel launch: ", c_cpu[1])
