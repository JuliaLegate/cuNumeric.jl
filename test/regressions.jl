using Test

@testset "regressions" begin
    @testset "array-size ABI preserves uint64_t" begin
        a = cuNumeric.zeros(UInt8, 1)
        try
            # Check the actual C API binding without a multi-GiB allocation.
            # An Int32 return truncates sizes above 2^31 - 1.
            actual = cuNumeric.nda_array_size(a)
            @test actual isa UInt64
            @test actual == length(a)
        finally
            cuNumeric.destroy!(a)
        end
    end
end
