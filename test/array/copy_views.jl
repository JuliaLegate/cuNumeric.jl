using Test, cuNumeric

@testset "copyto! follows Julia element order" begin
    allowscalar(false)
    begin
        for T in (Float32, Int32, ComplexF32, Bool)
            values = T === Bool ? Bool[isodd(i) for i in 1:24] : T.(1:24)
            for (sdims, ddims) in (((5, 5), (25,)), ((6,), (2, 3)),
                                  ((2, 3), (3, 2)), ((2, 3, 4), (4, 6)),
                                  ((), (1, 1)), ((1,), ()), ((0, 3), (0, 2)))
                n = prod(sdims)
                vals = n == 25 ? vcat(values, one(T)) : values[1:n]
                src = reshape(vals, sdims)
                expected = fill(zero(T), ddims)
                copyto!(expected, src)
                for input in (src, NDArray(src))
                    dest = NDArray(fill(zero(T), ddims))
                    @test (@inferred copyto!(dest, input)) === dest
                    @test Array(dest) == expected
                end
            end
        end

        host = reshape(Float32.(1:20), 4, 5)
        a = NDArray(host)
        dest = view(a, 2:3, 2:4)
        copyto!(view(host, 2:3, 2:4), Float32[31, 32, 33, 34, 35, 36])
        copyto!(dest, NDArray(Float32[31, 32, 33, 34, 35, 36]))
        @test Array(a) == host
        @test copyto!(a, a) === a
        for input in (ones(Float32, 7), ones(Float32, 5), Float32[])
            @test_throws DimensionMismatch copyto!(dest, input)
            @test_throws DimensionMismatch copyto!(dest, NDArray(input))
        end
        @test_throws DimensionMismatch (a[:, :] = cuNumeric.ones(Float32, 2))
        @test Array(a) == host

        # Whole-array indexed assignment still requires a complete RHS.
        a[:, :] = NDArray(Float32.(21:40))
        @test Array(a) == reshape(Float32.(21:40), 4, 5)

        # Overlapping views must read the original source values.
        for (dst, src) in ((2:6, 1:5), (1:5, 2:6))
            host = Float32.(1:6)
            a = NDArray(host)
            copyto!(view(host, dst), view(host, src))
            dest, source = view(a, dst), view(a, src)
            @test (@inferred copyto!(dest, source)) === dest
            @test Array(a) == host
        end
        host = reshape(Float32.(1:12), 3, 4)
        a = NDArray(host)
        copyto!(view(host, :, 2:3), view(host, 1:2, 1:3))
        dest, source = view(a, :, 2:3), view(a, 1:2, 1:3)
        @test (@inferred copyto!(dest, source)) === dest
        @test Array(a) == host
    end
end

@testset "Integer indices drop only indexed dimensions" begin
    allowscalar(false)
    begin
        for (dims, inds) in (((3, 4), (:, 2)), ((3, 4), (2, :)),
                             ((3, 4), (1:2, 2)), ((3, 4), (2, 2:3)),
                             ((1, 4), (:, 2)), ((3, 1), (2, :)),
                             ((3, 4), (:, 2:2)), ((0, 4), (:, 2)),
                             ((3, 4), (2, 3)), ((4,), (2,)),
                             ((2, 3, 4), (:, 2, :)), ((2, 1, 4), (1, :, 2:3)))
            host = reshape(Float32.(1:prod(dims)), dims)
            a = NDArray(host)
            expected = view(host, inds...)
            v = view(a, inds...)
            @test v isa NDArray{Float32,ndims(expected)}
            @test size(v) == size(expected)
            @test Array(v) == expected
            fill!(expected, 9f0)
            fill!(v, 9f0)
            @test Array(a) == host
            cuNumeric.destroy!(v)
            @test Array(a) == host
        end

        host = reshape(Float32.(1:12), 3, 4)
        a = NDArray(host)
        @test (@inferred view(a, :, 2)) isa NDArray{Float32,1}
        @test (@inferred view(a, 2, :)) isa NDArray{Float32,1}
        for inds in ((:, 2), (2, :), (1:2, 2), (2, 2:3))
            @test Array(a[inds...]) == host[inds...]
        end
        for dims in (1, 2)
            @test map(Array, collect(eachslice(a; dims))) == collect(eachslice(host; dims))
        end
        copyto!(view(a, :, 2), NDArray(Float32[21, 22, 23]))
        host[:, 2] .= Float32[21, 22, 23]
        copyto!(view(a, 2, :), NDArray(Float32[31, 32, 33, 34]))
        host[2, :] .= Float32[31, 32, 33, 34]
        @test Array(a) == host

        @accelerate function _copy_view_column!(a, x)
            a[:, 2] .= x
            a
        end
        @test _copy_view_column!(a, NDArray(Float32[41, 42, 43])) === a
        host[:, 2] .= Float32[41, 42, 43]
        @test Array(a) == host
        @test_throws BoundsError view(a, :, 5)
        nested = view(view(a, :, 2:3), :, 1)
        GC.gc(true)
        fill!(nested, 7f0)
        host[:, 2] .= 7f0
        @test Array(a) == host
    end
end
