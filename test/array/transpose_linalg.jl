using Test, LinearAlgebra, Random, cuNumeric

@testset "Lazy matrix transpose multiplication" begin
    cuNumeric.allowscalar(false)
    Random.seed!(735)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        for opA in (identity, transpose), opB in (identity, transpose)
            opA === identity && opB === identity && continue
            ah = randn(T, opA === identity ? (3, 5) : (5, 3))
            bh = randn(T, opB === identity ? (5, 2) : (2, 5))
            a, b = NDArray(ah), NDArray(bh)
            @test Array(opA(a) * opB(b)) ≈ opA(ah) * opB(bh)
            @test Array(a) == ah
            @test Array(b) == bh
        end
        ah, xh = randn(T, 5, 3), randn(T, 5)
        a, x = NDArray(ah), NDArray(xh)
        @test parent(transpose(a)) === a
        @test transpose(transpose(a)) === a
        @test Array(transpose(a) * x) ≈ transpose(ah) * xh
        @test Array(a) == ah
        @test Array(x) == xh
    end
end

@testset "Transpose multiplication delegates validation and empty inputs" begin
    cuNumeric.allowscalar(false)
    a = NDArray(ComplexF64[1+im 2-im; 3+2im 4+im])
    @test_throws DimensionMismatch transpose(a) * cuNumeric.ones(ComplexF64, 3, 2)
    @test_throws DimensionMismatch a * transpose(cuNumeric.ones(ComplexF64, 2, 3))
    @test_throws DimensionMismatch transpose(a) * transpose(cuNumeric.ones(ComplexF64, 2, 3))
    @test_throws DimensionMismatch transpose(a) * cuNumeric.ones(ComplexF64, 3)
    @test Array(a) == ComplexF64[1+im 2-im; 3+2im 4+im]
    empty_a = transpose(cuNumeric.zeros(ComplexF64, 0, 3))
    @test Array(empty_a * cuNumeric.zeros(ComplexF64, 0, 2)) == zeros(3, 2)
    @test Array(empty_a * cuNumeric.zeros(ComplexF64, 0)) == zeros(3)
    @test size(transpose(cuNumeric.zeros(Float64, 2, 0)) * cuNumeric.ones(Float64, 2, 3)) == (0, 3)
    @test size(transpose(cuNumeric.ones(Float64, 2, 3)) * cuNumeric.zeros(Float64, 2, 0)) == (3, 0)
    for T in (Bool, Int32)
        ai, bi = cuNumeric.ones(T, 2, 2), cuNumeric.ones(T, 2, 2)
        @test_throws ArgumentError transpose(ai) * bi
        @test_throws ArgumentError ai * transpose(bi)
        @test_throws ArgumentError transpose(ai) * cuNumeric.ones(T, 2)
    end
    a32, b64 = NDArray(Float32[1 2; 3 4]), NDArray([2.0 1; -1 3])
    cuNumeric.allowpromotion(false) do
        @test_throws "Implicit promotion" transpose(a32) * b64
    end
    cuNumeric.allowpromotion() do
        @test Array(transpose(a32) * b64) ≈ transpose(Float32[1 2; 3 4]) * [2.0 1; -1 3]
        x = NDArray([2.0, -1])
        @test Array(transpose(a32) * x) ≈ transpose(Float32[1 2; 3 4]) * [2.0, -1]
    end
end

@testset "Adjoint matrix multiplication is explicitly unsupported" begin
    cuNumeric.allowscalar(false)
    for T in (Float64, ComplexF64)
        a, b, x = cuNumeric.ones(T, 2, 2), cuNumeric.ones(T, 2, 2), cuNumeric.ones(T, 2)
        for (A, B) in ((a', b), (a, b'), (a', b'),
                       (transpose(a), b'), (a', transpose(b)), (a', x))
            @test_throws "Adjoint matrices is not supported" A * B
        end
        @test Array(a) == ones(T, 2, 2)
        @test Array(b) == ones(T, 2, 2)
    end
end
