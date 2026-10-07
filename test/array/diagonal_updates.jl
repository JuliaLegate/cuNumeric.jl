using Test, LinearAlgebra

@testset "Diagonal division writes existing destination storage" begin
    @allowpromotion for T in (Float32, Float64, ComplexF32, ComplexF64)
        dh = T <: Complex ? T[2 + im, 3 - im, 4 + 2im] : T[2, 3, 4]
        D = Diagonal(NDArray(dh))
        for ah in (T[2, 6, 12], reshape(T.(1:6), 3, 2))
            a = NDArray(ah)
            alias = view(a, ntuple(_ -> Colon(), ndims(a))...)
            expected = ah ./ (ndims(ah) == 1 ? dh : reshape(dh, 3, 1))
            @test Array(D \ a) ≈ expected
            @test ldiv!(D, a) === a
            @test Array(alias) ≈ expected
        end
        ah = reshape(T.(1:6), 2, 3)
        a = NDArray(ah)
        alias = view(a, :, :)
        expected = ah ./ reshape(dh, 1, 3)
        @test Array(a / D) ≈ expected
        @test rdiv!(a, D) === a
        @test Array(alias) ≈ expected
        @test Array(D.diag) == dh
    end

    # A shared diagonal must be read completely before its storage is overwritten.
    for divide! in (ldiv!, rdiv!)
        host = reshape(Float32.(1:9), 3, 3)
        a = NDArray(host)
        # NDArray slicing retains singleton dimensions; Diagonal needs a vector.
        D = Diagonal(cuNumeric.reshape(view(a, :, 1), (3,)))
        @test cuNumeric.nda_overlaps(a, D.diag)
        dh = copy(host[:, 1])
        expected = host ./ reshape(dh, divide! === ldiv! ? (3, 1) : (1, 3))
        divide! === ldiv! ? ldiv!(D, a) : rdiv!(a, D)
        @test Array(a) ≈ expected
        @test Array(D.diag) ≈ expected[:, 1]
    end
    parent = NDArray(Float32[2, 4, 8, 16])
    @test ldiv!(Diagonal(view(parent, 1:3)), view(parent, 2:4)) isa NDArray
    @test Array(parent) == Float32[2, 2, 2, 2]

    @allowpromotion for (AType, DType) in ((Float32, Float64), (Float64, Float32), (Int32, Float32))
        dh = DType[2, 4]
        ah = AType[4 8; 8 16]
        D = Diagonal(NDArray(dh))
        a, b = NDArray(ah), NDArray(ah)
        @test ldiv!(D, a) === a
        @test rdiv!(b, D) === b
        @test Array(a) == AType.(ah ./ reshape(dh, 2, 1))
        @test Array(b) == AType.(ah ./ reshape(dh, 1, 2))
    end

    for T in (Float32, Float64)
        dh = T[0, -0.0, Inf, NaN]
        ah = reshape(T[1, 0, -1, Inf, 0, 1, Inf, NaN], 2, 4)
        D = Diagonal(NDArray(dh))
        a = NDArray(ah)
        expected = ah ./ reshape(dh, 1, 4)
        @test isequal(Array(a / D), expected)
        @test rdiv!(a, D) === a
        @test isequal(Array(a), expected)
        b = NDArray(copy(permutedims(ah)))
        @test ldiv!(D, b) === b
        @test isequal(Array(b), permutedims(expected))
    end

    D = Diagonal(NDArray(Float32[2]))
    @test_throws DimensionMismatch ldiv!(D, cuNumeric.ones(Float32, 3))
    @test_throws DimensionMismatch ldiv!(D, cuNumeric.ones(Float32, 3, 2))
    @test_throws DimensionMismatch rdiv!(cuNumeric.ones(Float32, 2, 3), D)
    empty = Diagonal(NDArray(Float32[]))
    @test size(ldiv!(empty, cuNumeric.zeros(Float32, 0, 2))) == (0, 2)
    @test size(rdiv!(cuNumeric.zeros(Float32, 2, 0), empty)) == (2, 0)
end

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
        @allowfetch @test fetch(norm(D, p)) ≈ 5f0
        @test fetch(norm(D, fetch(p))) ≈ 5f0
    end
end

@testset "Diagonal condition results survive temporary cleanup" begin
    @allowpromotion for T in (Float32, Float64, ComplexF32, ComplexF64)
        dh = T <: Complex ? T[1 + im, 2 - im, 4 + im] : T[1, 2, 4]
        D = Diagonal(NDArray(dh))
        orders = (1, 2, Inf)
        results = map(p -> cond(D, p), orders)
        # Fetch only after temporary handles have been released and collected.
        GC.gc()
        cuNumeric.drain_pending_frees!()
        for (p, result) in zip(orders, results)
            @test fetch(result) ≈ cond(Diagonal(dh), p)
        end
        @test Array(D.diag) == dh
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
