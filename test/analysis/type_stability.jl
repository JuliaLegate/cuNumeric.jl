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
=#

@accelerate function _type_stable_accelerate_function(a, b)
    intermediate = @. a + b
    return @. intermediate * 2.0f0
end

function _type_stable_accelerate_begin(a, b)
    return @accelerate begin
        intermediate = @. a + b
        result = @. intermediate * 2.0f0
        (intermediate, result)
    end
end

function _type_stable_accelerate_let(a, b)
    return @accelerate let
        intermediate = @. a + b
        @. intermediate * 2.0f0
    end
end

function _type_stable_accelerate_expr(a, b)
    return @accelerate (@. (a + b) * 2.0f0)
end

_type_stable_cuda_argtypes(task::cuNumeric.CUDATask) = task.argtypes

@testset verbose = true "core" begin
    a = cuNumeric.zeros(5)
    b = cuNumeric.zeros(Float64, 3, 4)
    @test @inferred(size(a)) !== nothing
    @test @inferred(size(b)) !== nothing
    @test @inferred(cuNumeric.shape(a)) !== nothing
    @test @inferred(cuNumeric.shape(b)) !== nothing
end

@testset verbose = true "construction" begin
    # zeros, zeros_like, ones, rand, fill, trues, falses\
    for constructor in (:zeros, :ones)
        @eval begin
            @test @inferred(cuNumeric.$(constructor)(Float64, 3, 2)) !== nothing
            @test @inferred(cuNumeric.$(constructor)(Float64, (3, 4))) !== nothing
            @test @inferred(cuNumeric.$(constructor)(3, 5, 6)) !== nothing
            @test @inferred(cuNumeric.$(constructor)((3,))) !== nothing
            @test @inferred(cuNumeric.$(constructor)()) !== nothing
            @test @inferred(cuNumeric.$(constructor)(Int64)) !== nothing
        end
    end
    a = cuNumeric.zeros(Float64, 5, 3)
    @test @inferred(cuNumeric.zeros_like(a)) !== nothing

    for constructor in (:trues, :falses)
        @eval begin
            @test @inferred(cuNumeric.$(constructor)(5)) !== nothing
            @test @inferred(cuNumeric.$(constructor)((5, 4))) !== nothing
            @test @inferred(cuNumeric.$(constructor)(3, 4, 5)) !== nothing
        end
    end

    @test @inferred(cuNumeric.fill(2.0, 3, 4)) !== nothing
    @test @inferred(cuNumeric.fill(2, (3, 4))) !== nothing
    @test @inferred(cuNumeric.fill(2.0, 3)) !== nothing

    @test @inferred(cuNumeric.rand(4, 3)) !== nothing
    @test @inferred(cuNumeric.rand(Float32, 5)) !== nothing
    @test @inferred(cuNumeric.randn(Float64, 5)) !== nothing
    @test @inferred(cuNumeric.randexp(Float32, 5)) !== nothing
    @test @inferred(cuNumeric.rand(0:3, 3)) !== nothing
    @test @inferred(cuNumeric.rand(ComplexF32, 5)) !== nothing
    @test @inferred(cuNumeric.randn(ComplexF64, 5)) !== nothing

    # NDArray from Julia Array (Parent-stable attachment)
    @test @inferred(cuNumeric.NDArray(rand(10))) !== nothing
    @test @inferred(cuNumeric.NDArray(rand(Float32, 3, 3))) !== nothing
end

@testset verbose = true "custom CUDA metadata" begin
    task = cuNumeric.CUDATask("kernel", (Float32, Int32))
    @test isconcretetype(typeof(task))
    @test all(isconcretetype, fieldtypes(typeof(task)))
    @test @inferred(_type_stable_cuda_argtypes(task)) == DataType[Float32, Int32]

    storage_type = cuNumeric.PaddedStorage{Float32,1}
    @test all(isconcretetype, Base.uniontypes(fieldtype(storage_type, :backing)))
    @test all(isconcretetype, Base.uniontypes(fieldtype(storage_type, :staging)))
    @test all(isconcretetype, Base.uniontypes(fieldtype(storage_type, :shape)))
end

@testset verbose = true "conversion" begin
    # cast to array, as_type
    a = cuNumeric.zeros(Float64, 5, 5)
    @test @inferred(Array(a)) !== nothing
    @test @inferred(Array{Float32}(a)) !== nothing
    @test @inferred(cuNumeric.as_type(a, Float32)) !== nothing
    @test @inferred(cuNumeric.as_type(a, Int64)) !== nothing
end

@testset verbose = true "indexing" begin
    # getindex, setindex!, copy, copyto!, fill!, as_type
    a = cuNumeric.zeros(Float32, 5, 5)
    b = cuNumeric.zeros(Int32, 11)

    @test @inferred(a[1:3, 1:3]) !== nothing
    @test @inferred(a[2, 1:3]) !== nothing
    @test @inferred(a[1, 1:3] .+ b[1:3]) !== nothing
    @test @inferred(b[1:5]) !== nothing
    # @test @inferred(a[1:3, 1:end]) !== nothing
    allowscalar() do
        @test @inferred(a[1, 2]) !== nothing
    end
end

@testset verbose = true "broadcasting" begin
    a = cuNumeric.ones(Float32, 3, 3)
    b = cuNumeric.ones(Int32, 3, 3)
    @test @inferred(5 .* a) !== nothing
    @test @inferred(5.0f0 .* a) !== nothing
    @test @inferred(5 * a) !== nothing
    @test @inferred(5.0f0 * a) !== nothing

    @test @inferred(a .* b) !== nothing
    @test @inferred(a .+ b) !== nothing
    @test @inferred(a ./ b) !== nothing
    @test @inferred(((a .* b) .+ a) .* 2.0f0) !== nothing
end

@testset verbose = true "@accelerate forms" begin
    a = cuNumeric.ones(Float32, 3, 3)
    b = cuNumeric.ones(Float32, 3, 3)

    function_result = @inferred _type_stable_accelerate_function(a, b)
    @test function_result isa NDArray{Float32,2}

    begin_result = @inferred _type_stable_accelerate_begin(a, b)
    @test begin_result isa Tuple{NDArray{Float32,2},NDArray{Float32,2}}

    let_result = @inferred _type_stable_accelerate_let(a, b)
    @test let_result isa NDArray{Float32,2}

    expr_result = @inferred _type_stable_accelerate_expr(a, b)
    @test expr_result isa NDArray{Float32,2}
end

@testset verbose = true "solve" begin
    # native float/complex, 2D and 1D rhs
    @testset "$(T)" for T in Base.uniontypes(cuNumeric.SUPPORTED_SOLVE_TYPES)
        A = cuNumeric.NDArray(T[2 1; 5 7])
        b2 = cuNumeric.NDArray(T[11; 13;;]) # creates a 2d matrix instead of vector
        b1 = cuNumeric.NDArray(T[11, 13])
        @test @inferred(cuNumeric.solve(A, b2)) !== nothing
        @test @inferred(cuNumeric.solve(A, b1)) !== nothing
    end

    # int/bool promote to Float64 (under allowpromotion) and stay inferrable
    @testset "promote $(T)" for T in (Int32, Int64, Bool)
        A = cuNumeric.NDArray(T[1 0; 0 1])
        b = cuNumeric.NDArray(reshape(T[1, 1], 2, 1))
        allowpromotion() do
            @test @inferred(cuNumeric.solve(A, b)) !== nothing
        end
    end
end

@testset verbose = true "svd" begin
    @testset "$(T)" for T in Base.uniontypes(cuNumeric.SUPPORTED_SVD_TYPES)
        A = cuNumeric.NDArray(T[1 0; 0 1])
        @test @inferred(LinearAlgebra.svd(A)) !== nothing
        @test @inferred(LinearAlgebra.svd(A; full=true)) !== nothing
    end

    @testset "promote $(T)" for T in (Int32, Int64, Bool)
        A = cuNumeric.NDArray(T[1 0; 0 1])
        allowpromotion() do
            @test @inferred(LinearAlgebra.svd(A)) !== nothing
        end
    end
end

@testset verbose = true "qr" begin
    @testset "$(T)" for T in Base.uniontypes(cuNumeric.SUPPORTED_QR_TYPES)
        A = cuNumeric.NDArray(T[1 0; 0 1])
        @test @inferred(LinearAlgebra.qr(A)) !== nothing
    end

    @testset "promote $(T)" for T in (Int32, Int64, Bool)
        A = cuNumeric.NDArray(T[1 0; 0 1])
        allowpromotion() do
            @test @inferred(LinearAlgebra.qr(A)) !== nothing
        end
    end
end

@testset verbose = true "cholesky" begin
    @testset "$(T)" for T in Base.uniontypes(cuNumeric.SUPPORTED_CHOLESKY_TYPES)
        A = cuNumeric.NDArray(Matrix{T}(I, 2, 2))
        @test @inferred(LinearAlgebra.cholesky(A)) !== nothing
        B = cuNumeric.NDArray(reshape(Matrix{T}(I, 2, 2), 1, 2, 2))
        @test @inferred(cuNumeric.batched_cholesky(B)) !== nothing
    end

    @testset "promote $(T)" for T in (Int32, Int64, Bool)
        A = cuNumeric.NDArray(T[1 0; 0 1])
        allowpromotion() do
            @test @inferred(LinearAlgebra.cholesky(A)) !== nothing
        end
    end
end

# A synthetic configuration exercises MP selection without changing the live
# runtime cache or requiring multiple GPUs. Size remains a runtime argument.
function dl_mp_backend(op, shape)
    return cuNumeric._linalg_backend(op, shape, cuNumeric._LinalgRuntime(true, 4, 4))
end

@testset "distributed linear algebra inference" begin
    cn = cuNumeric
    mp = cn._CuSolverMpLinalg
    single = cn._SingleProcLinalg
    tiled = cn._TiledCholesky
    @test (@inferred cn._LinalgRuntime(true, 4, 4)).mp_eligible
    @test (@inferred cn._mp_row_partition(33, 4)) == (9, (4, 1))

    # Dynamic sizes legitimately infer a small union of backend tags, never Any.
    for (op, fallback, shape) in (
        (:solve, single, (cn.MIN_SOLVE_MATRIX_SIZE, cn.MIN_SOLVE_MATRIX_SIZE)),
        (:qr, single, (cn.MIN_QR_MATRIX_SIZE, 1)),
        (:cholesky, tiled, (cn.MIN_CHOLESKY_MATRIX_SIZE, cn.MIN_CHOLESKY_MATRIX_SIZE)),
    )
        @test dl_mp_backend(Val(op), shape) isa mp
        @test only(Base.return_types(dl_mp_backend, Tuple{Val{op},NTuple{2,Int}})) ==
            Union{mp,fallback}
    end

    # Infer the real MP launchers without executing collectives on this machine.
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        a = cn.NDArray{T,2,Nothing}
        @test only(Base.return_types(cn._solve!, Tuple{mp,a,a,a})) == a
        @test only(Base.return_types(cn._qr!, Tuple{mp,a,a,a})) == Nothing
        @test only(Base.return_types(cn._cholesky!, Tuple{mp,a,a})) == a
    end
end

@testset verbose = true "eigen" begin
    @testset "$(T)" for T in Base.uniontypes(cuNumeric.SUPPORTED_EIG_TYPES)
        A = cuNumeric.NDArray(Matrix{T}(I, 2, 2))
        @test @inferred(LinearAlgebra.eigen(A)) !== nothing
        @test @inferred(LinearAlgebra.eigvals(A)) !== nothing
        B = cuNumeric.NDArray(reshape(Matrix{T}(I, 2, 2), 1, 2, 2))
        @test @inferred(cuNumeric.batched_eigen(B)) !== nothing
        @test @inferred(cuNumeric.batched_eigvals(B)) !== nothing
    end

    @testset "promote $(T)" for T in (Int32, Int64, Bool)
        A = cuNumeric.NDArray(T[1 0; 0 1])
        allowpromotion() do
            @test @inferred(LinearAlgebra.eigen(A)) !== nothing
        end
    end
end

@testset verbose = true "fft" begin
    if cuNumeric._has_gpu_target()
        a = cuNumeric.zeros(ComplexF32, 8)
        b = cuNumeric.zeros(ComplexF32, 4, 6)
        @test @inferred(fft(a)) !== nothing
        @test @inferred(ifft(a)) !== nothing
        @test @inferred(fft(b, 1)) !== nothing
        @test @inferred(cuNumeric.batched_fft(b)) !== nothing
    end
end

@testset verbose = true "batched_solve" begin
    @testset "$(T)" for T in Base.uniontypes(cuNumeric.SUPPORTED_SOLVE_TYPES)
        A = cuNumeric.NDArray(reshape(T[2 1; 5 7], 1, 2, 2))
        b = cuNumeric.NDArray(reshape(T[11, 13], 1, 2, 1))
        @test @inferred(cuNumeric.batched_solve(A, b)) !== nothing
    end
end

@testset verbose = true "linalg ops" begin
    @testset "$(T)" for T in Base.uniontypes(cuNumeric.SUPPORTED_NUMERIC_TYPES)
        M = cuNumeric.zeros(T, 4, 3)
        sq = cuNumeric.zeros(T, 5, 5)
        v = cuNumeric.zeros(T, 8)
        @test @inferred(NDArray{T}(I, 5, 5)) !== nothing
        @test @inferred(cuNumeric.transpose(M)) !== nothing
        @test @inferred(cuNumeric.trace(sq)) !== nothing
        @test @inferred(cuNumeric.diag(sq)) !== nothing
    end
end

@testset verbose = true "sort" begin
    @testset "$(T)" for T in Base.uniontypes(cuNumeric.SUPPORTED_NUMERIC_TYPES)
        v = cuNumeric.zeros(T, 8)
        @test @inferred(cuNumeric.sort(v)) !== nothing
        if !(T <: Complex)
            @test @inferred(cuNumeric.searchsortedfirst(v, zero(T))) !== nothing
        end
    end
end

@testset verbose = true "unique" begin
    @testset "$(T)" for T in Base.uniontypes(cuNumeric.SUPPORTED_NUMERIC_TYPES)
        v = cuNumeric.zeros(T, 8)
        @test @inferred(cuNumeric.unique(v)) !== nothing
    end
end
