using Test, LinearAlgebra, cuNumeric

@testset "Identity mul! without promotion or scalar access" begin
    allowscalar(false)
    allowpromotion(false) do
        for T in (Float32, Float64, ComplexF32, Int32), dims in ((6,), (2, 3))
            host = reshape(T.(1:6), dims)
            src, dest = NDArray(host), cuNumeric.zeros(T, dims)
            @test mul!(dest, I, src) === dest
            @test Array(dest) == host
            @test mul!(dest, I, src, T(2), T(3)) === dest
            @test Array(dest) == 5host
            @test mul!(dest, I, dest) === dest
            @test Array(dest) == 5host
            @test mul!(dest, I, src, zero(T), zero(T)) === dest
            @test Array(dest) == zero(host)
            mul!(dest, zero(I), src)
            @test Array(dest) == zero(host)
            @test_throws DimensionMismatch mul!(cuNumeric.zeros(T, 7), I, NDArray(T.(1:6)))
            if length(dims) == 2
                @test mul!(dest, src, I) === dest
                @test Array(dest) == host
                @test mul!(dest, src, I, T(2), T(3)) === dest
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
        @test mul!(empty, I, empty) === empty

        parent = NDArray(Float32.(1:5))
        mul!(view(parent, 2:5), I, view(parent, 1:4))
        @test Array(parent) == Float32[1, 1, 2, 3, 4]
    end
end
