using Test, cuNumeric

@testset "NDArray inequality fallback" begin
    cases = (
        (Float32[1, 2], Float64[1, 2]),
        (Float32[1, 2], Float64[1, 3]),
        (Float64[NaN], Float64[NaN]),
        (Float64[-0.0], Float64[0.0]),
        (ComplexF64[1 + 2im], ComplexF64[1 + 3im]),
        (zeros(Float32, 2, 2), zeros(Float32, 4)),
        (Float32[], Float32[]),
    )
    allowfetch(false) do
        for (left, right) in cases
            a, b = NDArray(left), NDArray(right)
            result = a != b
            @test result isa CNBool
            @test fetch(result) === (left != right)
            @test_throws ArgumentError Bool(result)
            cuNumeric.destroy!(result)
            cuNumeric.destroy!(a)
            cuNumeric.destroy!(b)
        end
    end
end
