@testset "Julia broadcast axis alignment" begin
    allowscalar(false) do
        for T in (Float32, Float64)
            for dims in ((3, 3), (3, 4), (2, 3, 4), (2, 1, 3, 2))
                host = reshape(T.(1:prod(dims)), dims)
                bias = T.(10:10:(10 * first(dims)))
                a, b = NDArray(host), NDArray(bias)
                dest = similar(a)

                @test Array(a .+ b) == host .+ bias
                @test Array(b .- a) == bias .- host
                @test Array(T(2) .* a .+ b) == T(2) .* host .+ bias
                dest .= a .+ b
                @test Array(dest) == host .+ bias
                dest .= b
                @test Array(dest) == zero(host) .+ bias
                dest .= abs.(b)
                @test Array(dest) == zero(host) .+ abs.(bias)
                dest .= b .+ T(1) .+ T(2)
                @test Array(dest) == zero(host) .+ bias .+ T(3)

                # Reading the destination itself must still work in place.
                a .= a .+ b
                @test Array(a) == host .+ bias
            end

            # Both operand orders, and two array ranks within a 3D result.
            v = T[10, 20, 30]
            b, row = NDArray(v), NDArray(reshape(v, 1, 3))
            @test Array(b .+ row) == v .+ reshape(v, 1, 3)
            @test Array(row .- b) == reshape(v, 1, 3) .- v
            host3 = reshape(T.(1:24), 2, 3, 4)
            host2 = reshape(T.(10:10:60), 2, 3)
            @test Array(NDArray(host3) .+ NDArray(host2)) == host3 .+ host2

            # Assignment may drop excess trailing singleton axes as well as add them.
            dest = cuNumeric.zeros(T, 3)
            column = NDArray(reshape(v, 3, 1, 1))
            dest .= column .+ T(1)
            @test Array(dest) == v .+ T(1)
            dest .= column
            @test Array(dest) == v

            # A scalar store should not need a rank-normalizing view.
            scalar = NDArray(T(2))
            @test Array(b .+ scalar) == v .+ T(2)
            @test Array(scalar .- b) == T(2) .- v

            # Strided source/destination views and a source overlapping the destination.
            host = reshape(T.(1:30), 6, 5)
            a = NDArray(host)
            a[2:4, 2:4] .= b
            host[2:4, 2:4] .= v
            @test Array(a) == host
            source = view(a, 2:4, 2)
            @test Array(cuNumeric.zeros(T, 3, 4) .+ source) == zeros(T, 3, 4) .+ v
            a .= a .+ view(a, :, 1)
            host .= host .+ copy(view(host, :, 1))
            @test Array(a) == host

            empty_matrix = NDArray(zeros(T, 0, 4))
            @test size(empty_matrix .+ NDArray(T[])) == (0, 4)
            @test_throws DimensionMismatch b .+ cuNumeric.zeros(T, 2, 3)
            @test_throws DimensionMismatch dest .= cuNumeric.ones(T, 3, 2)
        end
    end
end
