using Test, LinearAlgebra, Random
import cuNumeric

dl_host(a) = cuNumeric.allowscalar() do
    Array(a)
end
dl_tol(::Type{T}) where {T} = 200 * eps(real(T))
function dl_record(op, T, residual)
    get(ENV, "CUNUMERIC_LINALG_VERBOSE", "0") == "1" &&
        println(op, " ", T, " residual: ", residual)
    return nothing
end

@testset "linear algebra task selection" begin
    cn = cuNumeric
    for (op, limit) in (
        (:solve, cn.MIN_SOLVE_MATRIX_SIZE), (:cholesky, cn.MIN_CHOLESKY_MATRIX_SIZE)
    )
        for available in (false, true), gpus in (0, 1, 2, 4), n in (limit - 1, limit, limit + 1)
            backend = cn._linalg_backend(Val(op), (n, n), available, gpus, max(2, gpus))
            @test (backend isa cn._CuSolverMpLinalg) == (available && gpus > 1 && n >= limit)
        end
    end
    qr_volumes = (
        cn.MIN_QR_MATRIX_SIZE - 1, cn.MIN_QR_MATRIX_SIZE, cn.MIN_QR_MATRIX_SIZE + 1
    )
    for available in (false, true), gpus in (0, 1, 2, 4), volume in qr_volumes
        backend = cn._linalg_backend(Val(:qr), (volume, 1), available, gpus, max(2, gpus))
        @test (backend isa cn._CuSolverMpLinalg) == (
            available && gpus > 1 && volume >= cn.MIN_QR_MATRIX_SIZE
        )
    end
    for op in (:solve, :cholesky)
        @test cn._linalg_backend(Val(op), (4, 9000, 9000), true, 4, 4) isa cn._SingleProcLinalg
    end
    @test cn._linalg_backend(
        Val(:cholesky), (9000, 9000), true, 4, 4; lower=false
    ) isa cn._SingleProcLinalg
    @test cn._linalg_backend(
        Val(:cholesky), (9000, 9000), true, 4, 4; inplace=true
    ) isa cn._SingleProcLinalg
    @test cn._linalg_backend(Val(:cholesky), (9000, 9000), true, 1, 1) isa cn._SingleProcLinalg
    @test cn._linalg_backend(Val(:cholesky), (9000, 9000), false, 4, 4) isa cn._TiledCholesky
    @test cn._linalg_backend(Val(:qr), (0, 10), true, 4, 4) isa cn._SingleProcLinalg
    @test cn._mp_row_partition(33, 4) == (9, (4, 1))
    @test cn._mp_row_partition(2, 4) == (1, (2, 1))
    @test cn._cholesky_color_shape(33, 4; min_matrix=0, min_tile=4) == (16, 16)
    @test cn._cholesky_color_shape(cn.MIN_CHOLESKY_MATRIX_SIZE, 4) == (1, 1)
end

function dl_check_solve(T; distributed=false)
    cn = cuNumeric
    rng = MersenneTwister(71)
    n = 33
    a = randn(rng, T, n, n) + T(n) * I
    da = cn.NDArray(a)
    for nrhs in (1, 3)
        b = randn(rng, T, n, nrhs)
        db = cn.NDArray(b)
        x = if distributed
            out = cn.zeros(T, n, nrhs)
            cn._solve!(cn._CuSolverMpLinalg(), out, da, db; tile=4)
        else
            da \ db
        end
        hx = dl_host(x)
        residual = norm(a * hx - b) / (norm(a) * norm(hx) + norm(b))
        dl_record(distributed ? "MP_SOLVE" : "solve", T, residual)
        @test residual <= dl_tol(T)
        @test dl_host(da) == a
        @test dl_host(db) == b
    end
    # Public vector-RHS reshape path, independent of the low-level MP tests.
    b = randn(rng, T, n)
    db = cn.NDArray(b)
    x = da \ db
    @test size(x) == (n,)
    @test isapprox(dl_host(x), a \ b; rtol=dl_tol(T))
end

function dl_check_cholesky(T; backend=nothing)
    cn = cuNumeric
    rng = MersenneTwister(72)
    n = 33
    z = randn(rng, T, n, n)
    a = z * z' + T(n) * I
    # Only the lower triangle is meaningful, including for complex input.
    input = copy(a)
    for j in 1:n, i in 1:(j - 1)
        input[i, j] = T(123)
    end
    da = cn.NDArray(input)
    factors = if backend === nothing
        f = cholesky(da)
        @test f isa Cholesky
        f.factors
    elseif backend isa cn._CuSolverMpLinalg
        cn._cholesky!(backend, cn.zeros(T, n, n), da; tile=4)
    else
        cn._cholesky!(backend, cn.zeros(T, n, n), da; min_matrix=0, min_tile=4)
    end
    l = dl_host(factors)
    residual = norm(a - l * l') / norm(a)
    dl_record("cholesky ($(typeof(backend)))", T, residual)
    @test residual <= dl_tol(T)
    @test istril(l)
    @test dl_host(da) == input
end

function dl_check_qr(T; distributed=false)
    cn = cuNumeric
    rng = MersenneTwister(73)
    for (m, n) in ((33, 33), (65, 17), (17, 65))
        a = randn(rng, T, m, n)
        da = cn.NDArray(a)
        q, r = if distributed
            cn._qr(cn._CuSolverMpLinalg(), da; tile=4)
        else
            f = qr(da)
            @test f isa cn.NDArrayQR
            f.Q, f.R
        end
        k = min(m, n)
        @test size(q) == (m, k)
        @test size(r) == (k, n)
        hq, hr = dl_host(q), dl_host(r)
        residual = norm(a - hq * hr) / norm(a)
        dl_record(distributed ? "MP_QR ($m, $n)" : "qr ($m, $n)", T, residual)
        @test residual <= dl_tol(T)
        @test norm(hq' * hq - I) / sqrt(k) <= dl_tol(T)
        @test istriu(hr)
        @test dl_host(da) == a
    end
end

@testset "distributed linear algebra numerics" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset "$T" begin
            dl_check_solve(T)
            dl_check_cholesky(T)
            dl_check_qr(T)
            dl_check_cholesky(T; backend=cuNumeric._TiledCholesky())
            if cuNumeric.cusolvermp_available() && cuNumeric.Legate.num_gpus() > 1
                dl_check_solve(T; distributed=true)
                dl_check_cholesky(T; backend=cuNumeric._CuSolverMpLinalg())
                dl_check_qr(T; distributed=true)
            else
                @test_skip "MP numerical tests require multiple active GPUs and cuSolverMp"
            end
        end
    end
end

@testset "empty and invalid solves" begin
    cn = cuNumeric
    @test size(cn.zeros(Float64, 0, 0) \ cn.zeros(Float64, 0)) == (0,)
    @test size(cn.zeros(Float64, 0, 0) \ cn.zeros(Float64, 0, 3)) == (0, 3)
    @test size(cn.zeros(Float64, 3, 3) \ cn.zeros(Float64, 3, 0)) == (3, 0)
    @test_throws ArgumentError cn.zeros(Float64, 2, 3) \ cn.zeros(Float64, 2)
    @test_throws ArgumentError cn.zeros(Float64, 3, 3) \ cn.zeros(Float64, 2)
    for (m, n) in ((0, 0), (0, 3), (3, 0))
        f = qr(cn.zeros(Float64, m, n))
        @test size(f.Q) == (m, min(m, n))
        @test size(f.R) == (min(m, n), n)
    end
end
