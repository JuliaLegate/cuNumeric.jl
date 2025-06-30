# found in examples/daxpy.jl
using cuNumeric

arr = cuNumeric.rand(NDArray, 20)

α = 1.32
b = 2.0

arr2 = α*arr + b

arr2[:] # disp array
