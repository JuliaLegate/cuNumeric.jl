using Test, LinearAlgebra, Random
import cuNumeric

dl_host(a) = cuNumeric.allowscalar() do
    Array(a)
end
dl_tol(::Type{T}) where {T} = 200 * eps(real(T))
dl_backend(op, shape, available, gpus, procs) =
    cuNumeric._linalg_backend(op, shape, cuNumeric._LinalgRuntime(available, gpus, procs))

@testset "linear algebra task selection" begin
    cn = cuNumeric
    for (op, limit) in (
        (:solve, cn.MIN_SOLVE_MATRIX_SIZE), (:cholesky, cn.MIN_CHOLESKY_MATRIX_SIZE)
    )
        for available in (false, true), gpus in (0, 1, 2, 4), n in (limit - 1, limit, limit + 1)
            backend = dl_backend(Val(op), (n, n), available, gpus, max(2, gpus))
            @test (backend isa cn._CuSolverMpLinalg) == (available && gpus > 1 && n >= limit)
        end
    end
    qr_volumes = (
        cn.MIN_QR_MATRIX_SIZE - 1, cn.MIN_QR_MATRIX_SIZE, cn.MIN_QR_MATRIX_SIZE + 1
    )
    for available in (false, true), gpus in (0, 1, 2, 4), volume in qr_volumes
        backend = dl_backend(Val(:qr), (volume, 1), available, gpus, max(2, gpus))
        @test (backend isa cn._CuSolverMpLinalg) == (
            available && gpus > 1 && volume >= cn.MIN_QR_MATRIX_SIZE
        )
    end
    for op in (:solve, :cholesky)
        @test dl_backend(Val(op), (4, 9000, 9000), true, 4, 4) isa cn._SingleProcLinalg
    end
    @test dl_backend(Val(:cholesky), (9000, 9000), true, 1, 1) isa cn._SingleProcLinalg
    @test dl_backend(Val(:cholesky), (9000, 9000), false, 4, 4) isa cn._TiledCholesky
    @test dl_backend(Val(:qr), (0, 10), true, 4, 4) isa cn._SingleProcLinalg
    @test cn._mp_row_partition(33, 4) == (9, (4, 1))
    @test cn._mp_row_partition(2, 4) == (1, (2, 1))
    n = max(cn.MIN_CHOLESKY_MATRIX_SIZE + 1, 100 * cn.MIN_CHOLESKY_TILE_SIZE)
    colors = cn._cholesky_color_shape(n, 4)
    @test colors[1] == colors[2]
    @test 4 <= colors[1] <= 4 * cn.MAX_CHOLESKY_TILES_PER_PROC
    @test cn._cholesky_color_shape(cn.MIN_CHOLESKY_MATRIX_SIZE, 4) == (1, 1)
end

@testset "cached runtime configuration" begin
    rt = cuNumeric._LINALG_RUNTIME[]
    @test rt.available == cuNumeric.cusolvermp_available()
    @test rt.gpus == Int(cuNumeric.Legate.num_gpus())
    @test rt.procs == Int(cuNumeric.Legate.num_procs())
    @test rt.mp_eligible == (rt.available && rt.gpus > 1)
    @test cuNumeric.choose_nd_color_shape((33, 33)) == (1, 1)
    @test cuNumeric.choose_nd_color_shape((5, 33, 33)) == (rt.procs, 1, 1)
    tiles, colors = cuNumeric.prepare_manual_task_for_batched_matrices((5, 33, 33))
    @test tiles == (cld(5, rt.procs), 33, 33)
    @test colors == (cld(5, tiles[1]), 1, 1)
end

function dl_check_solve(T)
    cn = cuNumeric
    rng = MersenneTwister(71)
    n = 33
    a = randn(rng, T, n, n) + T(n) * I
    da = cn.NDArray(a)
    for nrhs in (1, 3)
        b = randn(rng, T, n, nrhs)
        db = cn.NDArray(b)
        x = da \ db
        hx = dl_host(x)
        residual = norm(a * hx - b) / (norm(a) * norm(hx) + norm(b))
        @test residual <= dl_tol(T)
        @test dl_host(da) == a
        @test dl_host(db) == b
    end
    # Exercise the public vector-RHS reshape path on the selected backend.
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
    else
        cn._cholesky!(backend, cn.zeros(T, n, n), da)
    end
    l = dl_host(factors)
    residual = norm(a - l * l') / norm(a)
    @test residual <= dl_tol(T)
    @test istril(l)
    @test dl_host(da) == input
end

function dl_check_qr(T)
    cn = cuNumeric
    rng = MersenneTwister(73)
    @testset "QR shape ($m, $n)" for (m, n) in (
        (33, 33), (65, 17), (17, 65), (7, 1), (1, 7)
    )
        a = randn(rng, T, m, n)
        da = cn.NDArray(a)
        f = qr(da)
        @test f isa cn.NDArrayQR
        q, r = f.Q, f.R
        k = min(m, n)
        @test size(q) == (m, k)
        @test size(r) == (k, n)
        hq, hr = dl_host(q), dl_host(r)
        residual = norm(a - hq * hr) / norm(a)
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
    # Reject mismatched batches before constructing partitions, including empties.
    for (a_batch, b_batch) in ((2, 3), (3, 2), (0, 2), (2, 0))
        @test_throws "matching batch dimensions" cn.batched_solve(
            cn.zeros(Float64, a_batch, 3, 3), cn.zeros(Float64, b_batch, 3, 1)
        )
    end
    @test size(cn.batched_solve(cn.zeros(Float64, 0, 3, 3), cn.zeros(Float64, 0, 3, 1))) ==
        (0, 3, 1)
    for (m, n) in ((0, 0), (0, 3), (3, 0))
        f = qr(cn.zeros(Float64, m, n))
        @test size(f.Q) == (m, min(m, n))
        @test size(f.R) == (min(m, n), n)
    end
end
