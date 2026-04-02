# found in examples/daxpy.jl
using cuNumeric

arr = cuNumeric.rand(20)

α = 1.32f0
b = 2.0f0

arr2 = α .* arr .+ b

println(arr2)
