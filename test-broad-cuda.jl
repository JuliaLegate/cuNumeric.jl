using cuNumeric
using CUDA

function wrapped!(a, y, b)
    @inbounds begin
        a .= y .+ b
    end
    return nothing
end

N = 1024
threads = 256
blocks = cld(N, threads)

a = cuNumeric.zeros(Float32, N)
b = cuNumeric.full(N, 2.0f0)
y = cuNumeric.full(N, 1.0f0)

println(size(a), typeof(a))
println(size(b), typeof(b))
println(size(y), typeof(y))

task = cuNumeric.@cuda_task wrapped!(a, y, b)
cuNumeric.@launch task=task threads=threads blocks=blocks inputs=(b, y) outputs=a
