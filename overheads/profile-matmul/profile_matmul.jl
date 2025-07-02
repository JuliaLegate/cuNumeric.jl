using Pkg
Pkg.activate("../../cuNumeric.jl/pkg")
Pkg.instantiate()


using LinearAlgebra
using cuNumeric

FT = Float32
N = 10_000

A = cuNumeric.as_type(cuNumeric.rand(N,N), cuNumeric.LegateType(FT))
B = cuNumeric.as_type(cuNumeric.rand(N,N), cuNumeric.LegateType(FT))
C = cuNumeric.zeros(FT, N, N)

cuNumeric.mul!(C,A,B)

# Forces completion of mul!
println(C[1,1])

