include("cg.jl")
import Krylov, KrylovKit

cuNumeric.allowscalar(false)
cuNumeric.CUDACore.allowscalar(false)
for (backend, make) in (("NDArray", NDArray), ("CuArray", cuNumeric.CUDACore.CuArray))
    a, rhs = make(Ah), make(bh)
    for (name, solve) in (
        "Krylov CG" => () -> Krylov.cg(a, rhs; atol=0.0, rtol=1e-8, itmax=100),
        "Krylov CG (similar workspace)" => () -> begin
            workspace = Krylov.CgWorkspace(Krylov.KrylovConstructor(rhs))
            Krylov.cg!(workspace, a, rhs; atol=0.0, rtol=1e-8, itmax=100)
            workspace.x, workspace.stats
        end,
        "KrylovKit CG" => () -> KrylovKit.linsolve(a, rhs, zero(rhs), KrylovKit.CG(; tol=1e-8, maxiter=100)),
    )
        println("\n", backend, " / ", name)
        try
            solution, info = @allowautofetch solve()
            residual = norm(Ah * Array(solution) - bh) / norm(bh)
            println("Relative residual: ", residual)
            @assert residual <= 1e-8
        catch err
            showerror(stdout, err, catch_backtrace())
            println()
        end
        flush(stdout)
    end
end
