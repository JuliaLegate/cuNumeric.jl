include("cg.jl")

for (name, solve) in (
    "MINRES" => () -> IterativeSolvers.minres(A, b; reltol=1e-8, maxiter=100),
    "GMRES" => () -> IterativeSolvers.gmres(A, b; reltol=1e-8, maxiter=100),
    "BiCGStab(l)" => () -> IterativeSolvers.bicgstabl(A, b; reltol=1e-8, max_mv_products=100),
)
    println("\n", name)
    try
        solution = @allowautofetch solve()
        residual = norm(Ah * Array(solution) - bh) / norm(bh)
        println("Relative residual: ", residual)
        @assert residual <= 1e-8
    catch err
        showerror(stdout, err, catch_backtrace())
        println()
    end
    flush(stdout)
end
