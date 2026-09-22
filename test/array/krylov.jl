using Krylov

@testset "Krylov extension" begin
    @test Base.get_extension(cuNumeric, :cuNumericKrylovExt) !== nothing

    @testset "$T" for T in (Float32, Float64, ComplexF32, ComplexF64)
        R = real(T)
        hostx = T[1, 2, 3]
        hosty = T[4, 5, 6]
        # Complex coefficients exercise the complex scalar wrapper as well.
        a = T <: Complex ? T(2 + im) : T(2)
        b = T <: Complex ? T(3 - im) : T(3)
        α, β = sum(NDArray([a])), sum(NDArray([b]))
        x = NDArray(hostx)
        # Neither helpers nor coefficient dispatch should implicitly fetch.
        allowautofetch(false) do
            for s in (α, α.value)
                y = NDArray(copy(hosty))
                @test Krylov.kaxpy!(3, s, x, y) === y
                @test Array(y) ≈ a .* hostx .+ hosty
            end
            for s in (a, α, α.value), t in (b, β, β.value)
                y = NDArray(copy(hosty))
                @test Krylov.kaxpby!(3, s, x, t, y) === y
                @test Array(y) ≈ a .* hostx .+ b .* hosty
            end
        end

        n = 16
        offdiag = T <: Complex ? T(-1 + 0.25im) : T(-1)
        hostA = Matrix(Tridiagonal(fill(conj(offdiag), n - 1), fill(T(4), n), fill(offdiag, n - 1)))
        hostb = T.(1:n)
        A, rhs = NDArray(hostA), NDArray(hostb)
        workspace = Krylov.CgWorkspace(Krylov.KrylovConstructor(rhs))
        tol = 20 * eps(R)
        # Complex CG has real coefficients; permit their promotion to complex.
        @allowpromotion @allowautofetch Krylov.cg!(workspace, A, rhs; atol=zero(R), rtol=tol, itmax=100, history=true)
        @test workspace.stats.solved
        @test norm(hostA * Array(workspace.x) - hostb) / norm(hostb) <= 5tol
        @test !isempty(workspace.stats.residuals)
        solution, stats = @allowpromotion @allowautofetch Krylov.cg(A, rhs; atol=zero(R), rtol=tol, itmax=100)
        @test stats.solved
        @test solution isa NDArray
        @test norm(hostA * Array(solution) - hostb) / norm(hostb) <= 5tol
    end
end
