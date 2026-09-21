using cuNumeric, IterativeSolvers, LinearAlgebra, Random, Test
cuNumeric.allowscalar(false)
with_solver_promotion(f, ::Type{<:Real}) = f()
with_solver_promotion(f, ::Type{<:Complex}) = cuNumeric.allowpromotion(f)

@testset "Automatic NDArray CG extension" begin
    ext = Base.get_extension(cuNumeric, :cuNumericIterativeSolversExt)
    @test ext !== nothing
    @test ext.has_iterator_hooks
    Random.seed!(20260920)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        with_solver_promotion(T) do
            @testset "$T" begin
                R = real(T)
                n = 32
                M = randn(T, n, n)
                Ah = Matrix(M' * M + n * I)
                bh = randn(T, n)
                A, b = NDArray(Ah), NDArray(bh)
                tol = R === Float32 ? R(2e-5) : R(1e-10)
                for Pl in (IterativeSolvers.Identity(), Diagonal(NDArray(diag(Ah))))
                    x, h = cg(A, b; Pl, reltol=tol, maxiter=4n, log=true)
                    @test h.isconverged
                    @test norm(Ah * Array(x) - bh) <= 5tol * norm(bh)
                    @test Array(x) ≈ Ah \ bh rtol=5tol
                    @test length(h[:resnorm]) == h.iters
                    @test all(v -> v isa NDArray{R,0}, h[:resnorm])
                    @test all(isfinite, only.(h[:resnorm]))
                    # The extension supplies state, not another iteration method.
                    it = IterativeSolvers.cg_iterator!(zero(b), A, b, Pl; reltol=tol)
                    @test which(iterate, (typeof(it), Int)).module === IterativeSolvers
                    @test it.residual isa NDArray{R,0}
                    @test it.tol isa NDArray{R,0}
                    @test iterate(it) !== nothing
                    @test it isa ext.NDArrayCGIterable ? it.prev_residual isa NDArray{R,0} : it.ρ isa NDArray{T,0}
                    initial = randn(T, n)
                    guess = NDArray(initial)
                    result, h = cg!(guess, A, b; Pl, reltol=tol, maxiter=4n, log=true)
                    @test result === guess
                    @test h.isconverged
                    @test norm(Ah * Array(guess) - bh) <= 5tol * norm(Ah * initial - bh)
                    zx, zh = cg(A, cuNumeric.zeros(T, n); Pl, log=true)
                    @test zh.isconverged
                    @test zh.iters == 0
                    @test all(iszero, Array(zx))
                    _, limited = cg(A, b; Pl, maxiter=1, reltol=tol, log=true)
                    @test !limited.isconverged
                    @test limited.iters == 1
                    _, empty = cg(A, b; Pl, maxiter=0, log=true)
                    @test !empty.isconverged
                    @test empty.iters == 0
                    @test_throws ArgumentError cg(A, b; Pl, verbose=true)
                    # No logging path also uses the upstream driver.
                    @test Array(cg(A, b; Pl, reltol=tol, maxiter=4n)) ≈ Ah \ bh rtol=5tol
                end
                # The patched generic algorithm must still support host arrays.
                for Pl in (IterativeSolvers.Identity(), Diagonal(diag(Ah)))
                    xh, hh = cg(Ah, bh; Pl, reltol=tol, maxiter=4n, log=true)
                    @test hh.isconverged
                    @test norm(Ah*xh-bh) <= 5tol*norm(bh)
                    @test eltype(hh[:resnorm]) === Float64
                end
                println("Automatic extension $T passed")
                flush(stdout)
            end
        end
    end
end
