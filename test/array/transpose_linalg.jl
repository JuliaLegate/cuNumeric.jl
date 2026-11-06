@testset "Transpose and adjoint operations" begin
    Random.seed!(253)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset "$T" begin
            H, J = randn(T, 3, 4), randn(T, 4, 2)
            operand(wrap, host) = wrap === identity ? NDArray(host) : wrap(NDArray(Matrix(wrap(host))))
            α, β = T <: Complex ? (T(2+im), T(-3+im)) : (T(2), T(-3))
            for left in (identity, Base.transpose, adjoint), right in (identity, Base.transpose, adjoint)
                left === right === identity && continue
                A, B = operand(left, H), operand(right, J)
                @test Array(A * B) ≈ H * J
                C = cuNumeric.ones(T, 3, 2)
                @test mul!(C, A, B, α, β) === C
                @test Array(C) ≈ α * H * J .+ β
                fill!(C, T(NaN))
                mul!(C, A, B)
                @test Array(C) ≈ H * J
                allowfetch(false) do
                    mul!(C, A, B, sum(NDArray([α])), NDArray(β))
                    @test Array(C) ≈ (α + β) * H * J
                end
                @test_throws DimensionMismatch A * cuNumeric.ones(T, 5, 2)
            end
            for wrap in (Base.transpose, adjoint)
                for host in (H, randn(T, 4))
                    P = NDArray(host)
                    W = wrap(P)
                    expected = Array(wrap(host))
                    @test Array(W) == expected
                    @test Array{T}(W) == Matrix{T}(W) == expected
                    @test collect(W) isa Matrix{T}
                    @test collect(W) == collect(ComplexF64, W) == expected
                    C = copy(W)
                    @test Array(C) == expected
                    @test (C isa NDArray) == (ndims(host) == 2)
                    fill!(C isa NDArray ? C : parent(C), zero(T))
                    @test Array(P) == host
                    @test Array(NDArray(W)) == expected
                    dest = cuNumeric.zeros(T, size(W))
                    @test copyto!(dest, W) === dest
                    @test Array(dest) == expected
                    @test Array(W .+ one(T)) == expected .+ one(T)
                    @test Array(dest .+ W) == 2expected
                    op! = wrap === adjoint ? adjoint! : transpose!
                    @test op!(dest, P) === dest
                    @test Array(dest) == expected
                    @test copyto!(W, dest) === W
                    @test Array(P) == host
                    @test copyto!(W, copy(W)) === W
                    @test Array(P) == host
                    W .= W .+ one(T)
                    @test Array(W) == expected .+ one(T)
                    value = T <: Complex ? T(2+im) : T(2)
                    @test fill!(W, value) === W
                    @test Array(W) == fill(value, size(W))
                    W .= zero(T)
                    @test all(iszero, Array(P))
                end
                A = NDArray(H)
                v = randn(T, 3)
                @test Array(wrap(A) * NDArray(v)) ≈ wrap(H) * v
                @test fetch(wrap(NDArray(v)) * NDArray(v)) ≈ wrap(v) * v
                for matwrap in (identity, Base.transpose, adjoint)
                    B = operand(matwrap, H)
                    row = wrap(NDArray(v)) * B
                    @test typeof(row).name.wrapper === typeof(wrap(v)).name.wrapper
                    @test Array(row) ≈ wrap(v) * H
                end
                u = randn(T, 2)
                x, y = NDArray(v), wrap(NDArray(u))
                @test Array(x * y) ≈ v * wrap(u)
                C = cuNumeric.ones(T, 3, 2)
                mul!(C, x, y, α, β)
                @test Array(C) ≈ α * v * wrap(u) .+ β
                square = NDArray(H[1:3, 1:3])
                @test_throws ArgumentError mul!(square, wrap(square), square)
                copyto!(square, wrap(square))
                @test Array(square) == wrap(H[1:3, 1:3])
                bad = wrap(NDArray(fill(T(NaN), 4, 3)))
                fill!(C, T(NaN))
                mul!(C, bad, NDArray(J), zero(T), zero(T))
                @test all(iszero, Array(C))
                @test Array(wrap(cuNumeric.zeros(T, 0, 3)) * cuNumeric.zeros(T, 0, 2)) == zeros(T, 3, 2)
            end
        end
    end
    A32, B64 = NDArray(Float32[1 2; 3 4]), NDArray([1.0 2; 3 4])
    @test_throws "Implicit promotion" A32' * B64
    @allowpromotion @test Array(A32' * B64) ≈ [10.0 14; 14 20]
    @test_throws ArgumentError cuNumeric.ones(Int32, 2, 2)' * cuNumeric.ones(Int32, 2, 2)
    @accelerate function _wrapper_gram(A)
        B = copy(A)
        B' * B
    end
    @test Array(_wrapper_gram(B64)) ≈ [10.0 14; 14 20]
end
