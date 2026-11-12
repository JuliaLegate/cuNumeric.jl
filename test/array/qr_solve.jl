using Test, LinearAlgebra, Random, cuNumeric

@testset "QR solves" begin
    rng = MersenneTwister(42)
    allowscalar(false) do
        for T in (Float32, ComplexF64), m in (3, 5)
            A = randn(rng, T, m, 3)
            F = qr(NDArray(A), NoPivot())
            for dims in ((m,), (m, 2))
                B = randn(rng, T, dims)
                b = NDArray(B)
                expected = A \ B
                @test Array(@inferred(F \ b)) ≈ expected
                x = similar(b, T, size(expected))
                @test ldiv!(x, F, b) === x
                @test Array(x) ≈ expected
                @test Array(b) == B
            end
            @test Array(F.Q) * Array(F.R) ≈ A
            for a in (copy(NDArray(A)), view(NDArray(A), :, :))
                G = qr!(a, NoPivot())
                @test typeof(G) === typeof(F)
                @test Array(a) ≈ Array(G.Q)
                @test Array(G.Q) * Array(G.R) ≈ A
                @test (G.Q === a) == (a isa NDArray{T,2,Nothing})
            end
        end

        F = qr(NDArray(Float32[2 1; 1 3]))
        @test_throws r"ColumnNorm.*NoPivot" qr(F.Q, ColumnNorm())
        @test_throws r"ColumnNorm.*NoPivot" qr!(F.Q, ColumnNorm())
        b = NDArray(Float32[1, 2])
        @test ldiv!(b, F, b) === b
        @test Array(b) ≈ Float32[0.2, 0.6]
        @test eltype(F \ NDArray(Int64[1, 2])) === Float32
        backing = cuNumeric.zeros(Float32, 4)
        @test Array(ldiv!(view(backing, 2:3), F, NDArray(Float32[1, 2]))) ≈ Float32[0.2, 0.6]
        @test_throws DimensionMismatch ldiv!(cuNumeric.zeros(Float32, 3), F, b)
        @test_throws DimensionMismatch F \ cuNumeric.zeros(Float32, 3)
        @test_throws ArgumentError ldiv!(F.Q, F, F.R)
        @test_throws ArgumentError qr(cuNumeric.ones(Float32, 2, 3)) \ b
        @test_throws ArgumentError qr!(cuNumeric.ones(Float32, 2, 3))
        @test_throws ArgumentError qr!(NDArray([1 2; 3 4]))

        @test_throws "Implicit promotion" F \ NDArray(Float64[1, 2])
        @allowpromotion @test Array(F \ NDArray(Float64[1, 2])) ≈ [0.2, 0.6] rtol = 1e-6
        @test size(qr(cuNumeric.zeros(Float32, 3, 0)) \ cuNumeric.zeros(Float32, 3)) == (0,)
    end
end
