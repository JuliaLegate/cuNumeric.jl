# Opt-in: run each case in a separate process under an external timeout.
# A failed collective can poison its runtime.
using cuNumeric, LinearAlgebra, Test

length(ARGS) == 1 && only(ARGS) in ("solve", "cholesky", "tiled_cholesky") ||
    error("Usage: julia --project test/linalg_errors.jl solve|cholesky|tiled_cholesky")
op = only(ARGS)
a = cuNumeric.zeros(Float64, 33, 33)
b = cuNumeric.ones(Float64, 33, 1)
cuNumeric.versioninfo()
backend = op == "tiled_cholesky" ? cuNumeric._TiledCholesky() :
    cuNumeric._linalg_backend(Val(Symbol(op)), a)
println("Testing numerical failure with ", typeof(backend))

@testset "$op numerical failure" begin
    expected = op == "solve" ? r"singular"i : r"positive definite"i
    @test_throws expected begin
        out = if op == "solve"
            a \ b
        elseif op == "cholesky"
            cholesky(a).factors
        else
            cuNumeric._cholesky!(backend, similar(a), a)
        end
        cuNumeric.allowscalar() do
            Array(out) # Demand the result so deferred task failures surface.
        end
        cuNumeric.Legate.issue_execution_fence(true)
    end
end
