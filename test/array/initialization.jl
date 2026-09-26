using Test, cuNumeric

@testset "undef constructor types and ranks" begin
    for T in Base.uniontypes(cuNumeric.SUPPORTED_ARRAY_TYPES), N in 0:6
        dims = ntuple(_ -> 1, N)
        for a in (NDArray{T}(undef, dims), NDArray{T}(undef, dims...))
            @test eltype(a) === T
            @test size(a) == dims
            cuNumeric.destroy!(a)
        end
    end
end

@testset "uninitialized NDArray construction" begin
    a = NDArray{Float32}(undef, 2, 3)
    @test size(a) == (2, 3)
    fill!(a, 2f0)
    @test Array(a) == fill(2f0, 2, 3)

    b = NDArray{Float64}(undef, (3, 2))
    @test size(b) == (3, 2)
    fill!(b, 4.0)
    @test Array(b) == fill(4.0, 3, 2)

    scalar = NDArray{Int32}(undef)
    @test size(scalar) == ()
    fill!(scalar, Int32(7))
    @test Array(scalar)[] == 7

    scalar_like = similar(NDArray{Int32}, ())
    fill!(scalar_like, Int32(9))
    @test Array(scalar_like)[] == 9

    same = similar(a)
    @test size(same) == size(a)
    fill!(same, 3f0)
    @test Array(same) == fill(3f0, size(a))

    typed = similar(a, Float64, (3, 2))
    @test size(typed) == (3, 2)
    fill!(typed, 5.0)
    @test Array(typed) == fill(5.0, 3, 2)

    by_type = similar(NDArray{Float32}, (Base.OneTo(2), Base.OneTo(3)))
    @test size(by_type) == (2, 3)
    fill!(by_type, 6f0)
    @test Array(by_type) == fill(6f0, 2, 3)

    empty = similar(a, Float32, (0, 3))
    @test size(empty) == (0, 3)
    @test size(Array(empty)) == (0, 3)

    @test all(iszero, Array(cuNumeric.zeros(Float32, 2, 3)))
end
