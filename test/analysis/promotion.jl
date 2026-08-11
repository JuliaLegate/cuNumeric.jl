@testset "Type and Shape Promotion" begin
    N = 100
    cunumeric_arr1 = cuNumeric.zeros(Float64, N)
    cunumeric_arr3 = cuNumeric.zeros(Float32, N)
    cunumeric_int64 = cuNumeric.zeros(Int64, N)
    cunumeric_int32 = cuNumeric.zeros(Int32, N)
    cunumeric_arr5 = cuNumeric.zeros(Float64, N - 1, N - 1)
    @test_throws "Implicit promotion" cunumeric_arr3 .+ cunumeric_arr1
    @test_throws "Implicit promotion" map(+, cunumeric_arr3, cunumeric_arr1)
    @test_throws DimensionMismatch cunumeric_arr1 .+ cunumeric_arr5
    @test_throws DimensionMismatch cunumeric_arr1 ./ cunumeric_arr5

    allowscalar() do
        @test cuNumeric.compare(
            cunumeric_arr1, cunumeric_int64 .+ cunumeric_arr1, atol(Float64), rtol(Float64)
        )
        r1 = @allowpromotion cunumeric_arr3 .+ cunumeric_arr1
        r2 = @allowpromotion map(+, cunumeric_arr3, cunumeric_arr1)
        @test cuNumeric.compare(r1, r2, atol(Float64), rtol(Float64))
    end
end
