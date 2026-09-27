using Test

@testset "Full-colon views share storage and own their handles" begin
    for shape in ((4,), (2, 3), (2, 2, 3))
        a = cuNumeric.zeros(Float32, shape)
        inds = ntuple(_ -> Colon(), length(shape))
        v = view(a, inds...)
        @test v !== a
        @test size(v) == shape
        fill!(v, 3f0)
        @test Array(a) == fill(3f0, shape)
        fill!(a, 4f0)
        @test Array(v) == fill(4f0, shape)
        cuNumeric.destroy!(v)
        a[inds...] .= 5f0
        @test Array(a) == fill(5f0, shape)
        copied = a[inds...]
        fill!(copied, 9f0)
        @test Array(a) == fill(5f0, shape)
    end
    a = cuNumeric.zeros(Float32, 0)
    @test size(view(a, :)) == (0,)

    a = NDArray(2f0)
    v = view(a)
    @test v !== a
    @test size(v) == ()
    fill!(v, 3f0)
    @test fetch(a) == 3f0
    cuNumeric.destroy!(v)
    @test fetch(a) == 3f0

    @accelerate function _full_view_update!(a)
        a[:, :] .= 7f0
        a
    end
    a = cuNumeric.zeros(Float32, 2, 3)
    @test _full_view_update!(a) === a
    @test Array(a) == fill(7f0, 2, 3)
end

@testset "In-place broadcast preserves existing views" begin
    a = NDArray(Float32[1, 2, 3, 4])
    v = view(a, 1:2)
    a .+= 1f0
    @test Array(v) == Float32[2, 3]
    a .= a .* 2f0 .+ 1f0
    @test Array(v) == Float32[5, 7]

    reshaped = cuNumeric.reshape(a, (2, 2))
    reshaped .+= 1f0
    @test Array(a) == Float32[6, 8, 10, 12]
    @test Array(v) == Float32[6, 8]

    # Changing result type must copy back into the existing destination store.
    ints = NDArray(Int32[1, 2, 3, 4])
    a .= ints .+ Int32(1)
    @test Array(v) == Float32[2, 3]

    # Shifted inputs need a temporary, including identity broadcasts.
    a = NDArray(Float32[1, 2, 3, 4])
    a[2:4] .= a[1:3]
    @test Array(a) == Float32[1, 1, 2, 3]
    a[2:4] .= a[1:3] .+ 10f0
    @test Array(a) == Float32[1, 11, 11, 12]
end

@testset "Reject unsupported multidimensional linear indexing" begin
    a = NDArray(Float32[1 3 5; 2 4 6])
    @test_throws ArgumentError a[1:2]
    @test_throws ArgumentError a[:]
    @test_throws ArgumentError view(a, :)
    @test_throws ArgumentError (a[1:2] = NDArray(Float32[9, 9]))
    @test_throws ArgumentError (a[:] = cuNumeric.ones(Float32, 2, 3))
    @test_throws ArgumentError (a[:] .= 9f0)
    @test Array(a[1:2, 2:3]) == Float32[3 5; 4 6]
    @test Array(a[:, 2]) == reshape(Float32[3, 4], 2, 1)
end
