using Test

@testset "1D conversion ownership" begin
    for T in (Float32, ComplexF32)
        expected = T[1, 2, 3, 4]
        source = copy(expected)
        a = cuNumeric.NDArray(source)
        try
            # Construction must finish reading source before returning.
            fill!(source, T(99))
            source = nothing
            GC.gc(true)
            @test Array(a) == expected

            # The returned Julia vector must not alias the NDArray.
            converted = Array(a)
            converted[1] = T(77)
            @test Array(a) == expected

            # It must also survive explicit destruction of the source owner.
            survivor = Array(a)
            cuNumeric.destroy!(a)
            cuNumeric.issue_execution_fence(; block=true)
            GC.gc(true)
            @test survivor == expected
        finally
            cuNumeric.destroy!(a)
        end
    end
end

@testset "Singleton vector host conversion" begin
    # Runtime-created singletons can use scalar futures; attached input
    # vectors do not exercise the same storage representation.
    for (T, S, value) in ((Float32, Float64, -0f0),
                          (ComplexF32, ComplexF64, ComplexF32(Inf, 0)),
                          (Bool, Int32, true))
        a = cuNumeric.fill(value, (1,))
        try
            converted = Array(a)
            widened = Array{S}(a)
            @test isequal(converted, T[value])
            @test isequal(widened, S[value])
            converted[1] = zero(T)
            @test isequal(Array(a), T[value])
            cuNumeric.destroy!(a)
            cuNumeric.issue_execution_fence(; block=true)
            GC.gc(true)
            @test isequal(widened, S[value])
        finally
            cuNumeric.destroy!(a)
        end
    end
    a = cuNumeric.zeros(Float32, 0)
    try
        @test Array(a) == Float32[]
        @test Array{Float64}(a) == Float64[]
    finally
        cuNumeric.destroy!(a)
    end
end
