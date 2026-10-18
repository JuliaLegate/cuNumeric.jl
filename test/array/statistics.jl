using Test, cuNumeric
import Statistics

@testset "Statistics API" begin
    @test cuNumeric.mean === Statistics.mean
    @test cuNumeric.var === Statistics.var
    @test cuNumeric.std === Statistics.std
    cuNumeric.allowpromotion() do
        for T in (Float32, Float64, Int32)
            host = T[1 3 5; 2 4 8]
            input = NDArray(host)
            for dims in (:, 1, 2, (1, 2))
                actual = Statistics.mean(input; dims)
                expected = Statistics.mean(host; dims)
                @test (dims isa Colon ? fetch(actual) : Array(actual)) ≈ expected
                cuNumeric.destroy!(actual)
                for corrected in (false, true), f in (Statistics.var, Statistics.std)
                    actual = f(input; dims, corrected)
                    expected = f(host; dims, corrected)
                    @test (dims isa Colon ? fetch(actual) : Array(actual)) ≈ expected
                    cuNumeric.destroy!(actual)
                end
            end
            cuNumeric.destroy!(input)
        end
    end
end
