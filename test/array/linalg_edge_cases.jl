using Test, LinearAlgebra, Random
using cuNumeric: cuNumeric

le_host(a) = cuNumeric.allowscalar() do
    return Array(a)
end
le_tol(::Type{T}) where {T} = 200 * eps(real(T))
le_residual(a, b) = norm(a - b) / max(norm(a), one(real(eltype(a))))

# Keep the padded parent so we can check that the operation preserves both
# its input and the elements outside the view. Do not copy the view before
# passing it to the operation: the backend must handle its layout.
function le_input(a, layout)
    if layout == :slice
        m, n = size(a)
        parent = fill(eltype(a)(19), m + 2, n + 2)
        parent[2:(m + 1), 2:(n + 1)] = a
        dp = cuNumeric.NDArray(parent)
        return view(dp, 2:(m + 1), 2:(n + 1)), dp, parent
    elseif layout == :transpose
        parent = copy(transpose(a))
        dp = cuNumeric.NDArray(parent)
        return cuNumeric.transpose(dp), dp, parent
    end
    dp = cuNumeric.NDArray(a)
    return dp, dp, a
end

function le_qr(a, da)
    m, n = size(a)
    k = min(m, n)
    f = qr(da)
    q, r = le_host(f.Q), le_host(f.R)
    @test size(q) == (m, k)
    @test size(r) == (k, n)
    @test le_residual(a, q * r) <= le_tol(eltype(a))
    @test norm(q' * q - I) <= le_tol(eltype(a)) * k
    @test istriu(r)
end

function le_svd(a, da)
    m, n = size(a)
    for full in (false, true)
        f = svd(da; full)
        u, s, vt = le_host(f.U), le_host(f.S), le_host(f.Vt)
        @test size(u) == (m, full ? m : n)
        @test size(s) == (n,)
        @test size(vt) == (n, n)
        @test le_residual(a, u[:, 1:n] * Diagonal(s) * vt) <= le_tol(eltype(a))
        @test norm(u' * u - I) <= le_tol(eltype(a)) * size(u, 2)
        @test norm(vt * vt' - I) <= le_tol(eltype(a)) * n
        @test all(s .>= 0)
        @test issorted(s; rev=true)
        @test isapprox(s, svdvals(a); atol=le_tol(eltype(a)), rtol=le_tol(eltype(a)))
    end
end

function le_eigen(a, da)
    f = eigen(da)
    w, v = le_host(f.values), le_host(f.vectors)
    n = size(a, 1)
    @test size(w) == (n,)
    @test size(v) == (n, n)
    @test norm(a * v - v * Diagonal(w)) / max(norm(a) * norm(v), 1) <= le_tol(eltype(a))
    @test all(j -> isapprox(norm(v[:, j]), 1; atol=le_tol(eltype(a))), 1:n)
    # The fixtures are Hermitian, so their spectra are real, including repeated
    # zero eigenvalues. Sorting by real part avoids arbitrary eigenvector order.
    expected = eigvals(Hermitian(a))
    for values in (w, le_host(eigvals(da)))
        @test maximum(abs, imag.(values)) <= le_tol(eltype(a)) * max(norm(a), 1)
        @test isapprox(
            sort(real.(values)), expected; atol=le_tol(eltype(a)), rtol=le_tol(eltype(a))
        )
    end
end

@testset "linear algebra degenerate inputs" begin
    @testset "$T" for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset "QR/SVD $kind ($m, $n)" for kind in (:zero, :rank_one),
            (m, n) in ((4, 4), (6, 4), (4, 6))

            u = T.(1:m)
            T <: Complex && (u .+= im .* reverse(u))
            a = kind == :zero ? zeros(T, m, n) : u * transpose(T.(1:n))
            da = cuNumeric.NDArray(a)
            le_qr(a, da)
            m >= n && le_svd(a, da)
            @test le_host(da) == a
        end
        @testset "square $kind" for kind in (:zero, :rank_one)
            a = zeros(T, 4, 4)
            kind == :rank_one && (a[1, 1] = 3)
            da = cuNumeric.NDArray(a)
            le_eigen(a, da)
            @test le_host(da) == a
            # A zero pivot is exact here. Materialize inside @test_throws so
            # asynchronous task errors are observed by the assertion.
            for b in (ones(T, 4), ones(T, 4, 2))
                db = cuNumeric.NDArray(b)
                @test_throws "Singular matrix" le_host(da \ db)
                @test le_host(db) == b
                @test le_host(da) == a
            end
            @test_throws "Matrix is not positive definite" le_host(cholesky(da).factors)
            @test le_host(da) == a
        end
    end
end

@testset "linear algebra input layouts" begin
    @testset "$T $layout" for T in (Float32, Float64, ComplexF32, ComplexF64),
        layout in (:slice, :transpose)

        rng = MersenneTwister(81)
        @testset "QR/SVD ($m, $n)" for (m, n) in ((4, 4), (6, 4), (4, 6))
            a = randn(rng, T, m, n)
            da, dp, parent = le_input(a, layout)
            le_qr(a, da)
            m >= n && le_svd(a, da)
            @test le_host(dp) == parent
        end
        z = randn(rng, T, 4, 4)
        a = z * z' + T(4) * I
        da, dp, parent = le_input(a, layout)
        l = le_host(cholesky(da).factors)
        @test le_residual(a, l * l') <= le_tol(T)
        @test istril(l)
        @test le_host(dp) == parent
        le_eigen(a, da)
        @test le_host(dp) == parent
        # Also exercise a non-Hermitian solve so transpose/conjugation mistakes
        # cannot be hidden by the Cholesky/eigen fixture's symmetry.
        a = z + T(8) * I
        da, dp, parent = le_input(a, layout)
        @testset "solve $nrhs RHS" for nrhs in (1, 2)
            b = randn(rng, T, 4, nrhs)
            db, bp, bparent = le_input(b, layout)
            x = le_host(da \ db)
            @test size(x) == size(b)
            @test le_residual(b, a * x) <= le_tol(T)
            @test le_host(bp) == bparent
            @test le_host(dp) == parent
        end
        # A zero RHS is valid even though a zero coefficient matrix is not.
        for b in (zeros(T, 4), zeros(T, 4, 2))
            @test le_host(da \ cuNumeric.NDArray(b)) == b
        end
        @test le_host(dp) == parent
    end
end
