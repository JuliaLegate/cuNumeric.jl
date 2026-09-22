using cuNumeric, IterativeSolvers, LinearAlgebra

n = 32
Ah = Matrix(SymTridiagonal(fill(3.0, n), fill(-1.0, n - 1)))
bh = ones(n)
A, b = NDArray(Ah), NDArray(bh)

x = @allowautofetch IterativeSolvers.cg(A, b; reltol=1e-8, maxiter=100)
relative_residual = norm(Ah * Array(x) - bh) / norm(bh)
println("CG relative residual: ", relative_residual)
@assert relative_residual <= 1e-8
