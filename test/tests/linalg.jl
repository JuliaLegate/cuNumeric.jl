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

@testset "transpose" begin
    @testset verbose=true for T in Base.uniontypes(cuNumeric.SUPPORTED_NUMERIC_TYPES)
        A = my_rand(T, 4, 3)
        nda = cuNumeric.NDArray(A)

        ref = transpose(A)
        out = cuNumeric.transpose(nda)

        allowscalar() do
            @test cuNumeric.compare(ref, out, atol(T), rtol(T))
        end
    end
end

@testset "eye" begin
    @testset verbose=true for T in Base.uniontypes(cuNumeric.SUPPORTED_NUMERIC_TYPES)
        n = 5
        ref = Matrix{T}(I, n, n)
        out = cuNumeric.eye(T, n)
        allowscalar() do
            @test cuNumeric.compare(ref, out, atol(T), rtol(T))
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
            @test ref ≈ out[1] atol=atol(eltype(ref)) rtol=rtol(eltype(ref))
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
                @test ref ≈ out[1] atol=atol(eltype(ref)) rtol=rtol(eltype(ref))
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
                @test cuNumeric.compare(ref, out, atol(T), rtol(T))
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
#         @test cuNumeric.compare(ref, out, atol(Int32), rtol(Int32))
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
            @test cuNumeric.compare(fill(T(0.25), n, 1), x, atol(T), rtol(T))
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
            @test cuNumeric.compare(ref, x, atol(T), rtol(T))
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
            @test cuNumeric.compare(ref, x, atol(T), rtol(T))
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
            @test cuNumeric.compare(ref, x, atol(T), rtol(T))
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
                @test cuNumeric.compare(ref, x, atol(Float64), rtol(Float64))
            end
        end
    end
end


function check_svd_reconstruction(ref_A::AbstractMatrix, u, s, vh, tol_a, tol_r)
    U  = Array(u)
    S  = Array(s)
    Vh = Array(vh)
    A_rec = U * Diagonal(S) * Vh
    return isapprox(ref_A, A_rec; atol=tol_a, rtol=tol_r)
end

function check_svd_orthonormality(u, vh, tol_a, tol_r)
    U  = Array(u)
    Vh = Array(vh)
    ku = size(U, 2)
    kv = size(Vh, 1)
    ok_u  = isapprox(U'  * U,  Matrix{eltype(U)}(I, ku, ku);  atol=tol_a, rtol=tol_r)
    ok_vh = isapprox(Vh * Vh', Matrix{eltype(Vh)}(I, kv, kv); atol=tol_a, rtol=tol_r)
    return ok_u && ok_vh
end

@testset "svd square matrix" begin
    @testset verbose=true for T in Base.uniontypes(cuNumeric.SUPPORTED_SVD_TYPES)
        A_ref = my_rand(T, 5, 5)
        nda   = cuNumeric.NDArray(A_ref)
        u, s, vh = cuNumeric.svd(nda)
        allowscalar() do
            @test check_svd_reconstruction(A_ref, u, s, vh, atol(T), rtol(T))
            @test check_svd_orthonormality(u, vh, atol(T), rtol(T))
        end
    end
end

@testset "svd tall matrix (m > n)" begin
    @testset verbose=true for T in Base.uniontypes(cuNumeric.SUPPORTED_SVD_TYPES)
        A_ref = my_rand(T, 6, 4)
        nda   = cuNumeric.NDArray(A_ref)
        u, s, vh = cuNumeric.svd(nda, false)  # thin SVD for reconstruction test
        allowscalar() do
            @test check_svd_reconstruction(A_ref, u, s, vh, atol(T), rtol(T))
            @test check_svd_orthonormality(u, vh, atol(T), rtol(T))
        end
    end
end
 
@testset "svd thin output shapes (full_matrices=false)" begin
    @testset verbose=true for T in Base.uniontypes(cuNumeric.SUPPORTED_SVD_TYPES)
        m, n = 6, 4
        k     = min(m, n)
        A_ref = my_rand(T, m, n)
        nda   = cuNumeric.NDArray(A_ref)
        u, s, vh = cuNumeric.svd(nda, false)
        allowscalar() do
            @test size(Array(u))  == (m, k)
            @test size(Array(s))  == (k,)
            @test size(Array(vh)) == (k, n)
            @test check_svd_reconstruction(A_ref, u, s, vh, atol(T), rtol(T))
        end
    end
end
 
@testset "svd full output shapes (full_matrices=true)" begin
    @testset verbose=true for T in Base.uniontypes(cuNumeric.SUPPORTED_SVD_TYPES)
        m, n = 6, 4
        A_ref = my_rand(T, m, n)
        nda   = cuNumeric.NDArray(A_ref)
        u, s, vh = cuNumeric.svd(nda, true)
        allowscalar() do
            @test size(Array(u))  == (m, m)
            @test size(Array(s))  == (min(m, n),)
            @test size(Array(vh)) == (n, n)
        end
    end
end
 
@testset "svd singular values non-negative and sorted" begin
    @testset verbose=true for T in Base.uniontypes(cuNumeric.SUPPORTED_SVD_TYPES)
        A_ref = my_rand(T, 5, 5)
        nda   = cuNumeric.NDArray(A_ref)
        _, s, _ = cuNumeric.svd(nda)
        allowscalar() do
            sv = Array(s)
            @test all(sv .>= 0)
            @test issorted(sv; rev=true)
        end
    end
end
 
@testset "svd identity matrix" begin
    @testset verbose=true for T in Base.uniontypes(cuNumeric.SUPPORTED_SVD_TYPES)
        n     = 4
        A_ref = Matrix{T}(I, n, n)
        nda   = cuNumeric.NDArray(A_ref)
        _, s, _ = cuNumeric.svd(nda)
        allowscalar() do
            @test cuNumeric.compare(ones(T, n), s, atol(T), rtol(T))
        end
    end
end
 
@testset "svd rank-1 matrix" begin
    @testset verbose=true for T in Base.uniontypes(cuNumeric.SUPPORTED_SVD_TYPES)
        # outer product of two vectors: exactly one nonzero singular value
        # 5x4 satisfies the M >= N constraint
        v1    = T.(collect(1:5))
        v2    = T.(collect(1:4))
        A_ref = v1 * v2'
        nda   = cuNumeric.NDArray(A_ref)
        _, s, _ = cuNumeric.svd(nda)
        allowscalar() do
            sv = Array(s)
            @test sv[1] > atol(T)
            @test all(sv[2:end] .< sqrt(atol(T)) * 100)
        end
    end
end

@testset "svd promotion" begin
    @testset verbose=true for T in (Int32, Int64, Bool)
        vals = T == Bool ? T[1 0; 0 1] : reshape(T.(collect(1:4)), 2, 2)
        A = cuNumeric.NDArray(vals)
        @test_throws "Implicit promotion" cuNumeric.svd(A)
        allowpromotion() do
            u, s, vh = cuNumeric.svd(A)
            @test eltype(Array(u)) == Float64
        end
    end
end