#= Copyright 2026 Northwestern University,
 *                   Carnegie Mellon University University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author(s): David Krasowska <krasow@u.northwestern.edu>
 *            Ethan Meitz <emeitz@andrew.cmu.edu>
 *            Nader Rahal <naderrahhal2026@u.northwestern.edu>
=#

function gemm(N, M, T, max_diff)
    if T == Bool
        a = cuNumeric.trues(5, 5)
        b = cuNumeric.as_type(cuNumeric.trues(5, 5), Float32)
        c = cuNumeric.as_type(cuNumeric.trues(5, 5), Float64)
        @test_throws ArgumentError a * a # Bool * Bool not supported
        @allowpromotion d = a * b
        @allowpromotion e = a * c
        @test @allowscalar safe_compare(5 * ones(Float32, 5, 5), d, 0.0, max_diff)
        @test @allowscalar safe_compare(5 * ones(Float64, 5, 5), e, 0.0, max_diff)
        return nothing
    end

    if T <: Integer
        a = cuNumeric.ones(Int32, 5, 5)
        a_jl = ones(Int32, 5, 5)
        b = cuNumeric.ones(Float32, 5, 5)
        b_jl = ones(Float32, 5, 5)
        @test_throws ArgumentError a * a
        @test @allowscalar safe_compare(a_jl * b_jl, a * b, 0.0, max_diff)
        return nothing
    end

    dims_to_test = [(N, N), (N, M), (M, N)]

    @testset for dims in dims_to_test
        # Base julia arrays
        A_cpu = rand(T, dims[1], dims[2])
        B_cpu = rand(T, dims[2], dims[1])
        C_out_cpu = zeros(T, dims[1], dims[1])

        # cunumeric arrays
        A = cuNumeric.NDArray(A_cpu)
        B = cuNumeric.NDArray(B_cpu)
        C_out = cuNumeric.zeros(T, dims[1], dims[1])

        # Julia result
        C_cpu = A_cpu * B_cpu
        LinearAlgebra.mul!(C_out_cpu, A_cpu, B_cpu)

        @test C_cpu == C_out_cpu # really just making sure test is written right...

        A = cuNumeric.as_type(A, T)
        B = cuNumeric.as_type(B, T)
        C = cuNumeric.zeros(T, N, N)

        C = A * B
        LinearAlgebra.mul!(C_out, A, B)

        allowscalar() do
            @test isapprox(C, C_cpu, rtol=max_diff)
            @test isapprox(C, C_out, rtol=max_diff)

            if T != Float64
                C_wider = cuNumeric.zeros(Float64, dims[1], dims[1])
                @test_throws "Implicit promotion" LinearAlgebra.mul!(C_wider, A, B)
            end
        end

        # Integer output with FP input
        if !(T <: Integer)
            bad = cuNumeric.zeros(Int, dims[1], dims[1])
            @test_throws ArgumentError mul!(bad, A, B)
        end
    end
end

@testset "GEMM" begin
    N = 50
    M = 25
    @testset verbose = true for T in Base.uniontypes(cuNumeric.SUPPORTED_NUMERIC_TYPES)
        gemm(N, M, T, rtol(T))
    end
end

@testset "transpose" begin
    @testset verbose=true for T in Base.uniontypes(cuNumeric.SUPPORTED_NUMERIC_TYPES)
        A = my_rand(T, 4, 3)
        nda = cuNumeric.NDArray(A)

        ref = transpose(A)
        out = cuNumeric.transpose(nda)

        allowscalar() do
            @test safe_compare(ref, out, atol(T), rtol(T))
        end
    end
end

@testset "eye" begin
    @testset verbose=true for T in Base.uniontypes(cuNumeric.SUPPORTED_NUMERIC_TYPES)
        n = 5
        ref = Matrix{T}(I, n, n)
        out = NDArray{T}(I, n, n)
        allowscalar() do
            @test safe_compare(ref, out, atol(T), rtol(T))
        end
    end
end

@testset "trace" begin
    @testset verbose=true for T in Base.uniontypes(cuNumeric.SUPPORTED_NUMERIC_TYPES)
        A = my_rand(T, 6, 6)
        nda = cuNumeric.NDArray(A)

        ref = sum(diag(A))  # widens ints like trace's accumulator
        out = cuNumeric.trace(nda)
        allowscalar() do
            @test ref ≈ out[] atol=atol(eltype(ref)) rtol=rtol(eltype(ref))
        end
    end
end

@testset "trace with offset" begin
    @testset verbose=true for T in Base.uniontypes(cuNumeric.SUPPORTED_NUMERIC_TYPES)
        A = my_rand(T, 5, 5)
        nda = cuNumeric.NDArray(A)

        @testset "offset=$(k)" for k in (-2, -1, 0, 1, 2)
            ref = sum(diag(A, k))
            out = cuNumeric.trace(nda; offset=k)
            allowscalar() do
                @test ref ≈ out[] atol=atol(eltype(ref)) rtol=rtol(eltype(ref))
            end
        end
    end
end

@testset "diag" begin
    @testset verbose=true for T in Base.uniontypes(cuNumeric.SUPPORTED_NUMERIC_TYPES)
        A = my_rand(T, 6, 6)
        nda = cuNumeric.NDArray(A)

        @testset "k=$(k)" for k in (-2, 0, 3)
            ref = diag(A, k)
            out = cuNumeric.diag(nda; k=k)

            allowscalar() do
                @test safe_compare(ref, out, atol(T), rtol(T))
            end
        end
    end
end

# @testset "ravel" begin
#     A = reshape(collect(1:12), 3, 4)
#     nda = cuNumeric.NDArray(A)

#     ref = vec(A)
#     out = cuNumeric.ravel(nda)

#     allowscalar() do
#         @test safe_compare(ref, out, atol(Int32), rtol(Int32))
#     end
# end

@testset "unique" begin
    @testset verbose=true for T in Base.uniontypes(cuNumeric.SUPPORTED_NUMERIC_TYPES)
        A = T[1, 2, 2, 3, 4, 4, 4, 5]
        nda = cuNumeric.NDArray(A)

        ref = unique(A)
        out = cuNumeric.unique(nda)

        @test Set(Array(out)) == Set(ref)
    end
end

@testset "solve diagonal" begin
    @testset verbose=true for T in Base.uniontypes(cuNumeric.SUPPORTED_SOLVE_TYPES)
        n = 4
        A = cuNumeric.zeros(T, n, n)
        b = cuNumeric.zeros(T, n, 1)
        cuNumeric.@allowscalar for i in 1:n
            A[i, i] = T(4)
            b[i, 1] = T(1)
        end
        x = cuNumeric.solve(A, b)
        allowscalar() do
            @test safe_compare(fill(T(0.25), n, 1), x, atol(T), rtol(T))
        end
    end
end

@testset "solve identity" begin
    @testset verbose=true for T in Base.uniontypes(cuNumeric.SUPPORTED_SOLVE_TYPES)
        n = 4
        A = cuNumeric.NDArray(Matrix{T}(I, n, n))
        b = cuNumeric.NDArray(reshape(T.(collect(1:n)), n, 1))
        x = cuNumeric.solve(A, b)
        ref = reshape(T.(collect(1:n)), n, 1)
        allowscalar() do
            @test safe_compare(ref, x, atol(T), rtol(T))
        end
    end
end

@testset "solve general" begin
    @testset verbose=true for T in Base.uniontypes(cuNumeric.SUPPORTED_SOLVE_TYPES)
        A_ref = T[2 1; 5 7]
        b_ref = T[11; 13;;] # creates a 2d matrix instead of vector
        A = cuNumeric.NDArray(A_ref)
        b = cuNumeric.NDArray(b_ref)
        x = cuNumeric.solve(A, b)
        ref = A_ref \ b_ref
        allowscalar() do
            @test safe_compare(ref, x, atol(T), rtol(T))
        end
    end
end

@testset "solve vector rhs" begin
    @testset verbose=true for T in Base.uniontypes(cuNumeric.SUPPORTED_SOLVE_TYPES)
        A_ref = T[2 1; 5 7]
        b_ref = T[11, 13]
        x = cuNumeric.solve(cuNumeric.NDArray(A_ref), cuNumeric.NDArray(b_ref))
        @test ndims(x) == 1
        ref = A_ref \ b_ref
        allowscalar() do
            @test safe_compare(ref, x, atol(T), rtol(T))
        end
    end
end

@testset "solve promotion" begin
    @testset verbose=true for T in (Int32, Int64, Bool)
        A = cuNumeric.NDArray(T[1 0; 0 1])
        b = cuNumeric.NDArray(reshape(T[1, 1], 2, 1))

        # int/bool requires promotion to float. Will throw without allowpromtion()
        @test_throws "Implicit promotion" cuNumeric.solve(A, b)

        # ...allowed under @allowpromotion, result is Float64
        allowpromotion() do
            x = cuNumeric.solve(A, b)
            ref = Float64[1 0; 0 1] \ Float64[1; 1;;]
            allowscalar() do
                @test safe_compare(ref, x, atol(Float64), rtol(Float64))
            end
        end
    end
end

@testset "solve rejects batched input" begin
    A = cuNumeric.zeros(Float64, 2, 3, 3)
    b = cuNumeric.zeros(Float64, 2, 3, 1)
    @test_throws "batched_solve" cuNumeric.solve(A, b)
end

@testset "backslash" begin
    @testset verbose=true for T in Base.uniontypes(cuNumeric.SUPPORTED_SOLVE_TYPES)
        A_ref = T[2 1; 5 7]
        b_ref = T[11, 13]
        A = cuNumeric.NDArray(A_ref)

        x_vec = A \ cuNumeric.NDArray(b_ref)
        x_mat = A \ cuNumeric.NDArray(reshape(b_ref, 2, 1))

        allowscalar() do
            @test safe_compare(A_ref \ b_ref, x_vec, atol(T), rtol(T))
            @test safe_compare(A_ref \ reshape(b_ref, 2, 1), x_mat, atol(T), rtol(T))
        end
    end
end

function check_svd_reconstruction(ref_A::AbstractMatrix, F, tol_a, tol_r)
    A_rec = Array(F.U) * Diagonal(Array(F.S)) * Array(F.Vt)
    return isapprox(ref_A, A_rec; atol=tol_a, rtol=tol_r)
end

function check_svd_orthonormality(F, tol_a, tol_r)
    U = Array(F.U)
    Vt = Array(F.Vt)
    ku = size(U, 2)
    kv = size(Vt, 1)
    ok_u = isapprox(U' * U, Matrix{eltype(U)}(I, ku, ku); atol=tol_a, rtol=tol_r)
    ok_vt = isapprox(Vt * Vt', Matrix{eltype(Vt)}(I, kv, kv); atol=tol_a, rtol=tol_r)
    return ok_u && ok_vt
end

@testset "svd square matrix" begin
    @testset verbose=true for T in Base.uniontypes(cuNumeric.SUPPORTED_SVD_TYPES)
        A_ref = my_rand(T, 5, 5)
        F = LinearAlgebra.svd(cuNumeric.NDArray(A_ref))
        @test F isa LinearAlgebra.SVD
        allowscalar() do
            @test check_svd_reconstruction(A_ref, F, atol(T), rtol(T))
            @test check_svd_orthonormality(F, atol(T), rtol(T))
        end
    end
end

@testset "svd tall matrix (m > n)" begin
    @testset verbose=true for T in Base.uniontypes(cuNumeric.SUPPORTED_SVD_TYPES)
        A_ref = my_rand(T, 6, 4)
        F = LinearAlgebra.svd(cuNumeric.NDArray(A_ref))  # thin, for reconstruction
        allowscalar() do
            @test check_svd_reconstruction(A_ref, F, atol(T), rtol(T))
            @test check_svd_orthonormality(F, atol(T), rtol(T))
        end
    end
end

@testset "svd thin output shapes (full=false)" begin
    @testset verbose=true for T in Base.uniontypes(cuNumeric.SUPPORTED_SVD_TYPES)
        m, n = 6, 4
        k = min(m, n)
        A_ref = my_rand(T, m, n)
        F = LinearAlgebra.svd(cuNumeric.NDArray(A_ref))
        allowscalar() do
            @test size(Array(F.U)) == (m, k)
            @test size(Array(F.S)) == (k,)
            @test size(Array(F.Vt)) == (k, n)
            @test check_svd_reconstruction(A_ref, F, atol(T), rtol(T))
        end
    end
end

@testset "svd full output shapes (full=true)" begin
    @testset verbose=true for T in Base.uniontypes(cuNumeric.SUPPORTED_SVD_TYPES)
        m, n = 6, 4
        A_ref = my_rand(T, m, n)
        F = LinearAlgebra.svd(cuNumeric.NDArray(A_ref); full=true)
        allowscalar() do
            @test size(Array(F.U)) == (m, m)
            @test size(Array(F.S)) == (min(m, n),)
            @test size(Array(F.Vt)) == (n, n)
        end
    end
end

@testset "svd singular values non-negative and sorted" begin
    @testset verbose=true for T in Base.uniontypes(cuNumeric.SUPPORTED_SVD_TYPES)
        A_ref = my_rand(T, 5, 5)
        F = LinearAlgebra.svd(cuNumeric.NDArray(A_ref))
        allowscalar() do
            sv = Array(F.S)
            @test all(sv .>= 0)
            @test issorted(sv; rev=true)
        end
    end
end

@testset "svd identity matrix" begin
    @testset verbose=true for T in Base.uniontypes(cuNumeric.SUPPORTED_SVD_TYPES)
        n = 4
        A_ref = Matrix{T}(I, n, n)
        F = LinearAlgebra.svd(cuNumeric.NDArray(A_ref))
        allowscalar() do
            @test safe_compare(ones(T, n), F.S, atol(T), rtol(T))
        end
    end
end

@testset "svd rank-1 matrix" begin
    @testset verbose=true for T in Base.uniontypes(cuNumeric.SUPPORTED_SVD_TYPES)
        # outer product of two vectors: exactly one nonzero singular value
        # 5x4 satisfies the M >= N constraint
        v1 = T.(collect(1:5))
        v2 = T.(collect(1:4))
        A_ref = v1 * v2'
        F = LinearAlgebra.svd(cuNumeric.NDArray(A_ref))
        allowscalar() do
            sv = Array(F.S)
            @test sv[1] > atol(T)
            @test all(sv[2:end] .< sqrt(atol(T)) * 100)
        end
    end
end

@testset "svd promotion" begin
    @testset verbose=true for T in (Int32, Int64, Bool)
        vals = T == Bool ? T[1 0; 0 1] : reshape(T.(collect(1:4)), 2, 2)
        A = cuNumeric.NDArray(vals)
        @test_throws "Implicit promotion" LinearAlgebra.svd(A)
        allowpromotion() do
            F = LinearAlgebra.svd(A)
            @test eltype(Array(F.U)) == Float64
        end
    end
end

@testset "qr reconstruction" begin
    @testset verbose=true for T in Base.uniontypes(cuNumeric.SUPPORTED_QR_TYPES)
        A_ref = my_rand(T, 6, 4)
        q, r = LinearAlgebra.qr(cuNumeric.NDArray(A_ref))
        allowscalar() do
            Q = Array(q)
            R = Array(r)
            @test size(Q) == (6, 4)
            @test size(R) == (4, 4)
            @test isapprox(A_ref, Q * R; atol=atol(T), rtol=rtol(T))
            @test isapprox(Q' * Q, Matrix{eltype(Q)}(I, 4, 4); atol=atol(T), rtol=rtol(T))
        end
    end
end

@testset "qr square matrix" begin
    @testset verbose=true for T in Base.uniontypes(cuNumeric.SUPPORTED_QR_TYPES)
        A_ref = my_rand(T, 5, 5)
        q, r = LinearAlgebra.qr(cuNumeric.NDArray(A_ref))
        allowscalar() do
            Q = Array(q)
            R = Array(r)
            @test size(Q) == (5, 5)
            @test size(R) == (5, 5)
            @test isapprox(A_ref, Q * R; atol=atol(T), rtol=rtol(T))
        end
    end
end

@testset "qr promotion" begin
    @testset verbose=true for T in (Int32, Int64, Bool)
        vals = T == Bool ? T[1 0; 0 1] : reshape(T.(collect(1:4)), 2, 2)
        A = cuNumeric.NDArray(vals)
        @test_throws "Implicit promotion" LinearAlgebra.qr(A)
        allowpromotion() do
            q, r = LinearAlgebra.qr(A)
            allowscalar() do
                @test eltype(Array(q)) == Float64
                @test isapprox(
                    Float64.(vals), Array(q) * Array(r); atol=atol(Float64), rtol=rtol(Float64)
                )
            end
        end
    end
end

@testset "qr returns an NDArrayQR factorization" begin
    @testset verbose=true for T in Base.uniontypes(cuNumeric.SUPPORTED_QR_TYPES)
        A_ref = my_rand(T, 6, 4; L=real(T)(-1), R=real(T)(1))

        F = LinearAlgebra.qr(cuNumeric.NDArray(A_ref))
        @test F isa cuNumeric.NDArrayQR{T}
        @test F isa LinearAlgebra.Factorization{T}
        @test size(F) == (6, 4)
        @test size(F, 1) == 6
        @test size(F, 2) == 4
        @test size(F, 3) == 1
        @test sprint(show, MIME("text/plain"), F) isa String

        allowscalar() do
            @test isapprox(A_ref, Array(F.Q) * Array(F.R); atol=atol(T), rtol=rtol(T))
        end
    end
end

# Hermitian positive-definite, with a diagonal shift to keep it well conditioned.
function spd_matrix(::Type{T}, n) where {T}
    RT = real(T)
    B = my_rand(T, n, n; L=RT(-1), R=RT(1))
    return B * B' + T(n) * Matrix{T}(I, n, n)
end

@testset "cholesky reconstruction" begin
    @testset verbose=true for T in Base.uniontypes(cuNumeric.SUPPORTED_CHOLESKY_TYPES)
        n = 6
        A_ref = spd_matrix(T, n)
        F = LinearAlgebra.cholesky(cuNumeric.NDArray(A_ref))

        @test F isa LinearAlgebra.Cholesky
        @test F.uplo == 'L'

        allowscalar() do
            L = Array(F.factors)
            @test size(L) == (n, n)
            @test istril(L)  # `zeroout` clears the upper triangle in-task
            @test isapprox(A_ref, L * L'; atol=atol(T), rtol=rtol(T))
        end
    end
end

@testset "cholesky of the identity" begin
    @testset verbose=true for T in Base.uniontypes(cuNumeric.SUPPORTED_CHOLESKY_TYPES)
        n = 4
        A_ref = Matrix{T}(I, n, n)
        F = LinearAlgebra.cholesky(cuNumeric.NDArray(A_ref))
        allowscalar() do
            @test safe_compare(A_ref, F.factors, atol(T), rtol(T))
        end
    end
end

@testset "cholesky factorization object" begin
    n = 5
    T = Float64
    A_ref = spd_matrix(T, n)
    F = LinearAlgebra.cholesky(cuNumeric.NDArray(A_ref))

    L = F.L
    @test L isa LowerTriangular
    @test parent(L) === F.factors
    @test size(F) == (n, n)

    allowscalar() do
        @test isapprox(A_ref, Array(parent(L)) * Array(parent(L))'; atol=atol(T), rtol=rtol(T))
    end

    # `F.U` (and hence destructuring as `L, U = F`) needs `copy(F.factors')`,
    # which falls back to scalar indexing until `adjoint(::NDArray)` lands.
    @test_throws "scalar-indexed" F.U
end

@testset "cholesky rejects bad shapes and types" begin
    @test_throws ArgumentError LinearAlgebra.cholesky(cuNumeric.zeros(Float64, 3, 4))
    @test_throws ArgumentError LinearAlgebra.cholesky(cuNumeric.zeros(ComplexF64, 0, 0))
end

# The POTRF task raises this itself. It only surfaces as a catchable error
# because the launcher marks the task as throwing; without that Legate aborts
# the process. Not a PosDefException: the pivot index is not reported.
@testset "cholesky of a non-positive-definite matrix throws" begin
    A = cuNumeric.NDArray(Float64[1.0 2.0; 2.0 1.0])
    @test_throws "Matrix is not positive definite" begin
        F = LinearAlgebra.cholesky(A)
        allowscalar() do
            sum(abs.(Array(F.factors)))
        end
    end
end

@testset "cholesky promotion" begin
    @testset verbose=true for T in (Int32, Int64, Bool)
        vals = T == Bool ? T[1 0; 0 1] : T[2 0; 0 2]
        A = cuNumeric.NDArray(vals)
        @test_throws "Implicit promotion" LinearAlgebra.cholesky(A)
        allowpromotion() do
            F = LinearAlgebra.cholesky(A)
            allowscalar() do
                L = Array(F.factors)
                @test eltype(L) == Float64
                @test isapprox(Float64.(vals), L * L'; atol=atol(Float64), rtol=rtol(Float64))
            end
        end
    end
end

# Eigenvectors are only unique up to sign/phase, so compare the residual
# `A*v - λ*v` rather than the vectors themselves.
function eigen_residual(A_ref, values, vectors)
    C = eltype(values)
    return maximum(abs.(C.(A_ref) * vectors .- vectors * Diagonal(values)))
end

sort_spectrum(v) = sort(v; by=x -> (real(x), imag(x)))

@testset "eigen residual" begin
    @testset verbose=true for T in Base.uniontypes(cuNumeric.SUPPORTED_EIG_TYPES)
        n = 5
        A_ref = my_rand(T, n, n; L=real(T)(-1), R=real(T)(1))
        F = LinearAlgebra.eigen(cuNumeric.NDArray(A_ref))

        @test F isa LinearAlgebra.Eigen

        allowscalar() do
            values, vectors = Array(F.values), Array(F.vectors)
            # geev always produces complex output, even for a real input matrix
            @test eltype(values) == complex(T)
            @test eltype(vectors) == complex(T)
            @test size(values) == (n,)
            @test size(vectors) == (n, n)
            @test eigen_residual(A_ref, values, vectors) <= max(atol(T), rtol(T) * n)
            @test isapprox(
                sort_spectrum(values),
                sort_spectrum(LinearAlgebra.eigvals(A_ref)),
                atol=atol(T),
                rtol=rtol(T),
            )
        end
    end
end

@testset "eigen of a diagonal matrix" begin
    @testset verbose=true for T in Base.uniontypes(cuNumeric.SUPPORTED_EIG_TYPES)
        n = 4
        A_ref = Matrix{T}(Diagonal(T.(1:n)))
        values = LinearAlgebra.eigvals(cuNumeric.NDArray(A_ref))
        allowscalar() do
            @test isapprox(
                sort_spectrum(Array(values)),
                complex(T).(1:n),
                atol=atol(T),
                rtol=rtol(T),
            )
        end
    end
end

@testset "eigen factorization object" begin
    n = 5
    T = Float64
    A_ref = my_rand(T, n, n; L=-1.0, R=1.0)
    F = LinearAlgebra.eigen(cuNumeric.NDArray(A_ref))

    # an Eigen destructures as (values, vectors)
    values, vectors = F
    @test values === F.values
    @test vectors === F.vectors

    allowscalar() do
        @test eigen_residual(A_ref, Array(values), Array(vectors)) <= rtol(T) * n
    end
end

@testset "eigvals and eigvecs agree with eigen" begin
    n = 4
    T = Float64
    A_ref = my_rand(T, n, n; L=-1.0, R=1.0)
    nda = cuNumeric.NDArray(A_ref)

    values = LinearAlgebra.eigvals(nda)
    vectors = LinearAlgebra.eigvecs(nda)

    allowscalar() do
        @test eigen_residual(A_ref, Array(values), Array(vectors)) <= rtol(T) * n
    end
end

@testset "eigen rejects bad shapes and types" begin
    @test_throws ArgumentError LinearAlgebra.eigen(cuNumeric.zeros(Float64, 3, 4))
    @test_throws ArgumentError LinearAlgebra.eigvals(cuNumeric.zeros(Float64, 0, 0))
end

@testset "eigen promotion" begin
    @testset verbose=true for T in (Int32, Int64, Bool)
        vals = T == Bool ? T[1 0; 0 1] : T[2 0; 0 3]
        A = cuNumeric.NDArray(vals)
        @test_throws "Implicit promotion" LinearAlgebra.eigen(A)
        allowpromotion() do
            F = LinearAlgebra.eigen(A)
            allowscalar() do
                @test eltype(Array(F.values)) == ComplexF64
                @test isapprox(
                    sort_spectrum(Array(F.values)),
                    sort_spectrum(ComplexF64.(LinearAlgebra.eigvals(Float64.(vals)))),
                    atol=atol(Float64),
                    rtol=rtol(Float64),
                )
            end
        end
    end
end
