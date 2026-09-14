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
