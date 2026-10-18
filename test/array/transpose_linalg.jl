using Test, LinearAlgebra, Random, cuNumeric

@testset "Lazy matrix transpose and adjoint multiplication" begin
    cuNumeric.allowscalar(false)
    Random.seed!(735)
    ops = (identity, transpose, adjoint)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        for opA in ops, opB in ops
            opA === identity && opB === identity && continue
            @testset "$T / $opA / $opB" begin
                ah = randn(T, opA === identity ? (3, 5) : (5, 3))
                bh = randn(T, opB === identity ? (5, 2) : (2, 5))
                a, b = NDArray(ah), NDArray(bh)
                A, B = opA(a), opB(b)
                ref = opA(ah) * opB(bh)
                @test Array(A * B) ≈ ref
                c = cuNumeric.zeros(T, 3, 2)
                @test mul!(c, A, B) === c
                @test Array(c) ≈ ref
                seed = randn(T, 3, 2)
                copyto!(c, NDArray(seed))
                α = T <: Complex ? T(2 + im) : T(2)
                β = T <: Complex ? T(-3 + 2im) : T(-3)
                @test mul!(c, A, B, α, β) === c
                @test Array(c) ≈ α * ref + β * seed
                fill!(c, T(NaN))
                mul!(c, A, B, α, zero(T))
                @test Array(c) ≈ α * ref
                # Multiplication and conjugation must leave the parents intact.
                @test Array(a) == ah
                @test Array(b) == bh
            end
        end
        for op in (transpose, adjoint)
            ah, xh = randn(T, 5, 3), randn(T, 5)
            a, x = NDArray(ah), NDArray(xh)
            A = op(a)
            @test parent(A) === a
            @test op(A) === a
            @test Array(A * x) ≈ op(ah) * xh
            y = cuNumeric.ones(T, 3)
            @test mul!(y, A, x) === y
            @test Array(y) ≈ op(ah) * xh
            fill!(y, T(3))
            @test mul!(y, A, x, T(2), T(-1)) === y
            @test Array(y) ≈ 2 * op(ah) * xh .- 3
        end
    end
end

@testset "Wrapped multiplication coefficients, empty inputs and validation" begin
    cuNumeric.allowscalar(false)
    ah = ComplexF64[1+im 2-im; 3+2im 4+im]
    bh = ComplexF64[2-im 1+im; -1+im 3-2im]
    a, b = NDArray(ah), NDArray(bh)
    for op in (transpose, adjoint)
        A = op(a)
        for wrap in (identity, cuNumeric.cnscalar)
            c = cuNumeric.ones(ComplexF64, 2, 2)
            α, β = wrap(NDArray(2.0)), wrap(NDArray(-3.0))
            mul!(c, A, b, α, β)
            @test Array(c) ≈ 2 * op(ah) * bh .- 3
            y = cuNumeric.ones(ComplexF64, 2)
            mul!(y, A, NDArray(bh[:, 1]), α, β)
            @test Array(y) ≈ 2 * op(ah) * bh[:, 1] .- 3
        end
        @test_throws DimensionMismatch A * cuNumeric.ones(ComplexF64, 3, 2)
        @test_throws DimensionMismatch A * cuNumeric.ones(ComplexF64, 3)
        @test_throws DimensionMismatch mul!(cuNumeric.zeros(ComplexF64, 3, 2), A, b)
        @test_throws DimensionMismatch mul!(cuNumeric.zeros(ComplexF64, 3), A, NDArray(bh[:, 1]))
        @test_throws ArgumentError mul!(a, A, b)
        @test_throws ArgumentError mul!(b, A, b)
        @test_throws ArgumentError mul!(a, b, A)
        @test_throws ArgumentError mul!(view(a, :, :), A, b)
        @test_throws ArgumentError mul!(cuNumeric.zeros(Float64, 2, 2), A, b)
        x = NDArray(bh[:, 1])
        @test_throws ArgumentError mul!(x, A, x)

        bad = op(NDArray(fill(ComplexF64(NaN), 2, 2)))
        c = cuNumeric.ones(ComplexF64, 2, 2)
        mul!(c, bad, b, 0, 2)
        @test Array(c) == fill(2, 2, 2)
        mul!(c, bad, b, 0, 0)
        @test all(iszero, Array(c))
        @test_throws DimensionMismatch mul!(c, bad, cuNumeric.ones(ComplexF64, 3, 2), 0, 0)

        empty_a = op(cuNumeric.zeros(ComplexF64, 0, 3))
        @test Array(empty_a * cuNumeric.zeros(ComplexF64, 0, 2)) == zeros(3, 2)
        @test Array(empty_a * cuNumeric.zeros(ComplexF64, 0)) == zeros(3)
        @test size(op(cuNumeric.zeros(Float64, 2, 0)) * cuNumeric.ones(Float64, 2, 3)) == (0, 3)
        @test size(op(cuNumeric.ones(Float64, 2, 3)) * cuNumeric.zeros(Float64, 2, 0)) == (3, 0)
    end
    for op in (transpose, adjoint), T in (Bool, Int32)
        ai, bi = cuNumeric.ones(T, 2, 2), cuNumeric.ones(T, 2, 2)
        @test_throws ArgumentError op(ai) * bi
        @test_throws ArgumentError ai * op(bi)
        @test_throws ArgumentError mul!(cuNumeric.zeros(Float64, 2, 2), op(ai), bi, 0, 0)
    end
    a32, b64 = NDArray(Float32[1 2; 3 4]), NDArray([2.0 1; -1 3])
    cuNumeric.allowpromotion(false) do
        @test_throws "Implicit promotion" transpose(a32) * b64
    end
    cuNumeric.allowpromotion() do
        @test Array(transpose(a32) * b64) ≈ transpose(Float32[1 2; 3 4]) * [2.0 1; -1 3]
        @test_throws ArgumentError mul!(cuNumeric.zeros(Float32, 2, 2), transpose(a32), b64)
    end
end
