using Krylov

@testset "Krylov extension loading" begin
    @test Base.get_extension(cuNumeric, :cuNumericKrylovExt) !== nothing
end
