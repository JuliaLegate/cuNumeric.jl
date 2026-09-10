# Run through run.sh so that runtime configuration precedes package loading.
using cuNumeric, LinearAlgebra, Test

expected = parse(Int, ENV["CUNUMERIC_LINALG_EXPECT_GPUS"])
actual = Int(cuNumeric.Legate.num_gpus())
println("Julia: ", VERSION)
cuNumeric.versioninfo()
println("LEGATE_CONFIG: ", get(ENV, "LEGATE_CONFIG", ""))
println("Active GPUs: ", actual, "; expected: ", expected)
println("Active processors: ", cuNumeric.Legate.num_procs())
println("cuSolverMp available: ", cuNumeric.cusolvermp_available())
actual == expected || error("Runtime GPU count does not match the requested topology")
expected > 1 && !cuNumeric.cusolvermp_available() && error("cuSolverMp is unavailable")

# Includes all four element types, public APIs, forced small MP launches, and
# tiled Cholesky. No production policy constants are changed by these tests.
include(joinpath(@__DIR__, "../../test/array/distributed_linalg.jl"))

if "--production" in ARGS
    @testset "public APIs at production cutoffs" begin
        cn = cuNumeric
        n = cn.MIN_SOLVE_MATRIX_SIZE
        a = cn.NDArray{Float64}(I, n, n)
        println("Production solve backend: ", typeof(cn._linalg_backend(Val(:solve), a)))
        b = cn.ones(Float64, n)
        x = a \ b
        residual = norm(dl_host(x) .- 1) / sqrt(n)
        println("Production vector solve relative error: ", residual)
        @test residual <= dl_tol(Float64)

        n = cn.MIN_CHOLESKY_MATRIX_SIZE
        a = cn.NDArray{Float64}(I, n, n)
        println("Production Cholesky backend: ", typeof(cn._linalg_backend(Val(:cholesky), a)))
        f = cholesky(a)
        # Smoke-test sampled columns at this large cutoff; the small tests above
        # check complete reconstructions on nontrivial real/complex matrices.
        for cols in (1:4, (n - 3):n)
            got = dl_host(copy(f.factors[:, cols]))
            expected_columns = zeros(Float64, n, 4)
            for (j, i) in enumerate(cols)
                expected_columns[i, j] = 1
            end
            residual = norm(got - expected_columns)
            println("Production Cholesky sampled-column error: ", residual)
            @test residual <= dl_tol(Float64)
        end

        n = isqrt(cn.MIN_QR_MATRIX_SIZE)
        m = cld(cn.MIN_QR_MATRIX_SIZE, n)
        a = cn.NDArray{Float64}(I, m, n)
        println("Production QR backend: ", typeof(cn._linalg_backend(Val(:qr), a)))
        f = qr(a)
        q, r = dl_host(f.Q), dl_host(f.R)
        residual = norm(q * r - Matrix{Float64}(I, m, n)) / sqrt(n)
        println("Production QR relative reconstruction error: ", residual)
        @test residual <= dl_tol(Float64)
    end
end

cuNumeric.Legate.issue_execution_fence(true)
println("LINALG_ACCEPTANCE_PASSED")
println("Inspect the profile for MP_SOLVE, MP_POTRF, and MP_QR across the requested GPUs/nodes.")
