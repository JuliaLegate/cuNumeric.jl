using Test, LinearAlgebra, cuNumeric

@testset "Identity mul! without promotion or scalar access" begin
    allowscalar(false)
    allowpromotion(false) do
        for T in (Float32, Float64, ComplexF32, Int32), dims in ((6,), (2, 3))
            host = reshape(T.(1:6), dims)
            src, dest = NDArray(host), cuNumeric.zeros(T, dims)
            @test (@inferred mul!(dest, I, src)) === dest
            @test Array(dest) == host
            @test (@inferred mul!(dest, I, src, T(2), T(3))) === dest
            @test Array(dest) == 5host
            @test (@inferred mul!(dest, I, dest)) === dest
            @test Array(dest) == 5host
            @test (@inferred mul!(dest, I, src, zero(T), zero(T))) === dest
            @test Array(dest) == zero(host)
            mul!(dest, zero(I), src)
            @test Array(dest) == zero(host)
            @test_throws DimensionMismatch mul!(cuNumeric.zeros(T, 7), I, NDArray(T.(1:6)))
            if length(dims) == 2
                @test (@inferred mul!(dest, src, I)) === dest
                @test Array(dest) == host
                @test (@inferred mul!(dest, src, I, T(2), T(3))) === dest
                @test Array(dest) == 5host
            end
        end
        src = NDArray(Float32[1, 2, 3])
        dest = NDArray(fill(Float32(NaN), 3))
        mul!(dest, I, src)
        @test Array(dest) == Float32[1, 2, 3]
        fill!(src, Float32(NaN))
        mul!(dest, I, src, 0f0, 2f0)
        @test Array(dest) == Float32[2, 4, 6]
        mul!(dest, zero(I), src)
        @test Array(dest) == zeros(Float32, 3)
        empty = cuNumeric.zeros(Float32, 0)
        @test (@inferred mul!(empty, I, empty)) === empty

        parent = NDArray(Float32.(1:5))
        mul!(view(parent, 2:5), I, view(parent, 1:4))
        @test Array(parent) == Float32[1, 1, 2, 3, 4]
    end
end

@testset "Identity helper inference across coefficient branches" begin
    src = NDArray(Float32[1, 2, 3])
    dest = similar(src)
    a, b = cuNumeric.cnscalar(NDArray(2f0)), cuNumeric.cnscalar(NDArray(3f0))
    allowfetch(false) do
        allowpromotion(false) do
            for (J, α, β, expected) in (
                (I, 0f0, 0f0, Float32[0, 0, 0]),
                (I, 0f0, 3f0, Float32[12, 12, 12]),
                (zero(I), 2f0, 3f0, Float32[12, 12, 12]),
                (I, 1f0, 0f0, Float32[1, 2, 3]),
                (I, 2f0, 0f0, Float32[2, 4, 6]),
                (I, 2f0, 3f0, Float32[14, 16, 18]),
                (I, a, 0f0, Float32[2, 4, 6]),
                (I, 0f0, b, Float32[12, 12, 12]),
                (I, a, b, Float32[14, 16, 18]),
            )
                fill!(dest, 4f0)
                @test (@inferred cuNumeric._mul_identity!(dest, J, src, α, β)) === dest
                @test Array(dest) == expected
            end
            fill!(dest, 4f0)
            @test (@inferred mul!(dest, I, src, a, b)) === dest
            @test Array(dest) == Float32[14, 16, 18]
        end
    end
end
