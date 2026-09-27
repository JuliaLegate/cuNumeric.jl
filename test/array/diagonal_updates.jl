using Test, LinearAlgebra

@testset "Diagonal products write existing destination storage" begin
    @allowpromotion for T in (Float32, Float64, ComplexF32, ComplexF64)
        dh = T[2, 3, 4]
        D = Diagonal(NDArray(dh))
        for ah in (T[1, 2, 3], reshape(T.(1:6), 3, 2))
            a = NDArray(ah)
            c = cuNumeric.zeros(T, size(ah))
            v = view(c, ntuple(_ -> Colon(), ndims(c))...)
            @test mul!(c, D, a) === c
            @test Array(v) ≈ Diagonal(dh) * ah
            @test lmul!(D, a) === a
            @test Array(a) ≈ Diagonal(dh) * ah
        end
        ah = reshape(T.(1:6), 2, 3)
        a = NDArray(ah)
        c = cuNumeric.zeros(T, 2, 3)
        v = view(c, :, :)
        @test mul!(c, a, D) === c
        @test Array(v) ≈ ah * Diagonal(dh)
        @test rmul!(a, D) === a
        @test Array(a) ≈ ah * Diagonal(dh)
        @test_throws DimensionMismatch mul!(cuNumeric.zeros(T, 2), D, NDArray(T[1, 2, 3]))
        @test_throws DimensionMismatch mul!(cuNumeric.zeros(T, 2, 2), cuNumeric.ones(T, 2, 2), D)
    end

    # Partially overlapping vector input and output must use a temporary.
    a = NDArray(Float32[1, 2, 3, 4])
    D = Diagonal(NDArray(Float32[2, 3, 4]))
    mul!(view(a, 2:4), D, view(a, 1:3))
    @test Array(a) == Float32[1, 2, 6, 12]

    @allowpromotion begin
        c = cuNumeric.zeros(Float64, 3)
        mul!(c, D, NDArray(Float32[1, 2, 3]))
        @test Array(c) == [2.0, 6.0, 12.0]
    end
end

@testset "Diagonal norm dispatch and autofetch policy" begin
    dh = Float32[3, 4]
    D = Diagonal(NDArray(dh))
    for p in (0, 0f0, -0.0, 1, 1f0, 2, 2f0, 2.0, 3, 1.5, Inf, Inf32, -Inf32)
        actual = fetch(norm(D, p))
        expected = norm(Diagonal(dh), p)
        @test actual ≈ expected
        @test typeof(actual) == typeof(expected)
    end
    for p in (NDArray(2), cnscalar(NDArray(2)))
        @test_throws "Implicit CNScalar host extraction is disabled" norm(D, p)
        @allowautofetch @test fetch(norm(D, p)) ≈ 5f0
        @test fetch(norm(D, fetch(p))) ≈ 5f0
    end
end

@testset "Diagonal zero and negative norms" begin
    @allowpromotion for T in (Float32, Float64, ComplexF32, ComplexF64)
        for dh in (T[], T[2], T[0], T[2, 3], T[2, 0], T[NaN, 2], T[Inf, 2])
            D = Diagonal(NDArray(dh))
            for p in (0, -1, -2, -Inf)
                expected = norm(Diagonal(dh), p)
                actual = fetch(norm(D, p))
                @test isapprox(actual, expected; nans=true)
                @test typeof(actual) == typeof(expected)
            end
        end
    end
end

@testset "Diagonal product includes structural zeros" begin
    @allowpromotion for T in (Float32, Float64, ComplexF32, ComplexF64)
        for dh in (T[], T[2], T[0], T[2, 3], T[NaN, 2], T[Inf, 2],
                   T[floatmax(real(T)), floatmax(real(T))])
            @test isequal(fetch(prod(Diagonal(NDArray(dh)))), prod(Diagonal(dh)))
        end
    end
end
