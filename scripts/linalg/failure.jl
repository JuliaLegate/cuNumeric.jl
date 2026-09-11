# Keep each failure in its own process; a failed collective may poison a runtime.
using cuNumeric, LinearAlgebra

length(ARGS) == 2 || error("Usage: failure.jl solve|cholesky single|mp|tiled")
op, backend = ARGS
op in ("solve", "cholesky") || error("Unknown operation")
backend in ("single", "mp", "tiled") || error("Unknown backend")
op == "solve" && backend == "tiled" && error("No tiled solve backend")
cn = cuNumeric
backend == "mp" && cn._check_mp_launch(4)
expected_gpus = parse(Int, ENV["CUNUMERIC_LINALG_EXPECT_GPUS"])
Int(cn.Legate.num_gpus()) == expected_gpus || error("Unexpected runtime GPU count")
n = 33
a = cn.NDArray(zeros(Float64, n, n))
b = cn.ones(Float64, n, 1)
out = cn.zeros(Float64, n, op == "solve" ? 1 : n)
cn.Legate.issue_execution_fence(true)
expected_message = op == "solve" ? "singular" : "positive definite"

try
    if op == "solve"
        if backend == "mp"
            cn._solve!(cn._CuSolverMpLinalg(), out, a, b; tile=4)
        else
            cn._solve!(cn._SingleProcLinalg(), out, a, b)
        end
    elseif backend == "mp"
        cn._cholesky!(cn._CuSolverMpLinalg(), out, a; tile=4)
    elseif backend == "tiled"
        cn._cholesky!(cn._TiledCholesky(), out, a; min_matrix=0, min_tile=4)
    else
        cn._cholesky!(cn._SingleProcLinalg(), out, a)
    end
    cn.allowscalar() do
        Array(out) # Demand the failed result; errors may be deferred.
    end
    cn.Legate.issue_execution_fence(true)
catch err
    message = sprint(showerror, err)
    occursin(expected_message, lowercase(message)) || rethrow()
    println("EXPECTED_LINALG_FAILURE: ", message)
    exit(0)
end
error("The invalid input unexpectedly succeeded")
