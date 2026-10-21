using Krylov

@testset "Krylov GPU integration" begin
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
        allowfetch(false) do
            for s in (α, α.value)
                y = NDArray(copy(hosty))
                @test Krylov.kaxpy!(3, s, x, y) === y
                @test Array(y) ≈ a .* hostx .+ hosty
                @test Krylov.kdivcopy!(3, y, x, s) === y
                @test Array(y) ≈ hostx ./ a
                @test Krylov.kscal!(3, s, y) === y
                @test Array(y) ≈ hostx
                @test Krylov.kdiv!(3, y, s) === y
                @test Array(y) ≈ hostx ./ a
            end
            for s in (a, α, α.value), t in (b, β, β.value)
                y = NDArray(copy(hosty))
                @test Krylov.kaxpby!(3, s, x, t, y) === y
                @test Array(y) ≈ a .* hostx .+ b .* hosty
            end
            @test_throws ArgumentError Krylov.sym_givens(α, b)
        end
        for (s, t) in ((α, b), (a, β), (α, β))
            actual = @allowfetch Krylov.sym_givens(s, t)
            @test all(isapprox.(actual, Krylov.sym_givens(a, b)))
        end

        n = 16
        offdiag = T <: Complex ? T(-1 + 0.25im) : T(-1)
        hostA = Matrix(Tridiagonal(fill(conj(offdiag), n - 1), fill(T(4), n), fill(offdiag, n - 1)))
        hostb = T.(1:n)
        A, rhs = NDArray(hostA), NDArray(hostb)
        workspace = Krylov.CgWorkspace(Krylov.KrylovConstructor(rhs))
        tol = 20 * eps(R)
        # Complex CG has real coefficients; permit their promotion to complex.
        @allowpromotion @allowfetch Krylov.cg!(workspace, A, rhs; atol=zero(R), rtol=tol, itmax=100, history=true)
        @test workspace.stats.solved
        @test norm(hostA * Array(workspace.x) - hostb) / norm(hostb) <= 5tol
        @test !isempty(workspace.stats.residuals)
        solution, stats = @allowpromotion @allowfetch Krylov.cg(A, rhs; atol=zero(R), rtol=tol, itmax=100)
        @test stats.solved
        @test solution isa NDArray
        @test norm(hostA * Array(solution) - hostb) / norm(hostb) <= 5tol

        # BiCGSTAB exercises the same device-scalar hooks on a nonsymmetric operator.
        nonsymmetric = Matrix(Tridiagonal(fill(T(-0.3), n - 1), fill(T(4), n), fill(T(-0.8), n - 1)))
        B = NDArray(nonsymmetric)
        biworkspace = Krylov.BicgstabWorkspace(B, rhs)
        @allowpromotion @allowfetch Krylov.bicgstab!(biworkspace, B, rhs; atol=zero(R), rtol=tol, itmax=100)
        @test biworkspace.stats.solved
        @test norm(nonsymmetric * Array(biworkspace.x) - hostb) / norm(hostb) <= 5tol

        indefinite = hostA - Diagonal(T[iseven(i) ? 8 : 0 for i in 1:n])
        rectangular = vcat(hostA, Matrix{T}(I, 4, n))
        for (method, W, H) in ((:gmres, Krylov.GmresWorkspace, nonsymmetric),
                              (:minres, Krylov.MinresWorkspace, indefinite),
                              (:lsqr, Krylov.LsqrWorkspace, rectangular),
                              (:lsmr, Krylov.LsmrWorkspace, rectangular))
            @testset "$method" begin
                hb = T.(1:size(H, 1))
                op, rhs = NDArray(H), NDArray(hb)
                tol = R == Float32 ? R(1e-5) : R(1e-10)
                least_squares = method in (:lsqr, :lsmr)
                options = least_squares ? (; axtol=tol, btol=tol, etol=zero(R)) : (; atol=zero(R), rtol=tol)
                allocation = method == :gmres ? (; memory=3) : (; window=3)
                @allowpromotion @allowfetch begin
                    solution, stats = getproperty(Krylov, method)(op, rhs; options..., allocation..., itmax=150)
                    @test stats.solved
                    @test Array(solution) ≈ H \ hb rtol=50tol
                    workspace = W(op, rhs; allocation...)
                    preconditioner = Diagonal(NDArray(fill(T(0.25), length(hb))))
                    restart = method == :gmres ? (; restart=true) : (;)
                    for scale in (one(T), T(2))
                        copyto!(rhs, NDArray(scale .* hb))
                        getproperty(Krylov, Symbol(method, :!))(workspace, op, rhs;
                            options..., restart..., M=preconditioner, itmax=150, history=true)
                        @test workspace.stats.solved
                        @test Array(workspace.x) ≈ H \ (scale .* hb) rtol=50tol
                        @test !isempty(workspace.stats.residuals)
                    end
                end
            end
        end
    end
end
