using Test, LinearAlgebra, Random, cuNumeric, IterativeSolvers

@testset "Unmodified IterativeSolvers with NDScalar" begin
    @test pkgversion(IterativeSolvers) == v"0.9.4"
    cuNumeric.allowscalar(false)
    autounwrap(false)
    Random.seed!(20260920)
    cuNumeric.allowpromotion() do
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            R = real(T)
            M = randn(T, 16, 16)
            Ah = Matrix(M' * M + 16I)
            bh = randn(T, 16)
            A, b = NDArray(Ah), NDArray(bh)
            tol = R === Float32 ? R(2e-5) : R(1e-10)
            @test norm(b) isa NDReal{R}
            @test dot(b,b) isa NDScalar
            @test_throws ArgumentError cg(A,b; reltol=tol)
            for Pl in (IterativeSolvers.Identity(), Diagonal(NDArray(diag(Ah))))
                x, h = @autounwrap cg(A,b; Pl, reltol=tol, maxiter=64, log=true)
                @test h.isconverged
                @test norm(Ah*Array(x)-bh) <= 5tol*norm(bh)
                @test Array(x) ≈ Ah\bh rtol=5tol
                @test eltype(h[:resnorm]) === Float64
                initial = randn(T,16)
                x = NDArray(initial)
                result, h = autounwrap() do
                    cg!(x,A,b; Pl, reltol=tol, maxiter=64, log=true)
                end
                @test result === x
                @test h.isconverged
                @test norm(Ah*Array(x)-bh) <= 5tol*norm(Ah*initial-bh)
                z, h = @autounwrap cg(A,cuNumeric.zeros(T,16); Pl, log=true)
                @test h.isconverged && h.iters == 0
                @test all(iszero,Array(z))
            end
            @test_throws ArgumentError Float64(norm(b))
            println("Stock CG/PCG $T passed")
            flush(stdout)
        end
    end
end
