using Test, LinearAlgebra, Random, cuNumeric

la_value(x::cuNumeric.CNScalar) = fetch(x)

@testset "NDArray norms" begin
    cuNumeric.allowscalar(false)
    Random.seed!(912)
    for T in Base.uniontypes(Union{cuNumeric.SUPPORTED_FLOAT_TYPES,cuNumeric.SUPPORTED_COMPLEX_TYPES})
        @testset "$T" begin
            R = real(T)
            ah, xh = randn(T, 7, 5), randn(T, 5)
            a, x = cuNumeric.NDArray(ah), cuNumeric.NDArray(xh)
            empty = cuNumeric.zeros(T, 0)
            @test la_value(norm(empty)) === zero(R)
            @test la_value(norm(cuNumeric.zeros(T, 5))) === zero(R)
            for p in (0, 1, 2, 3, 0.5, -1, -2, Inf, -Inf)
                @test la_value(norm(x, p)) ≈ norm(xh, p)
                @test la_value(norm(a, p)) ≈ norm(ah, p)
            end
            @test norm(x) isa cuNumeric.CNReal{R}
            @test la_value(norm(cuNumeric.NDArray(T[0, 2, 0, 3]), 0)) == R(2)
            @test la_value(norm(cuNumeric.NDArray(T[0, 2]), -1)) == zero(R)
            @test isnan(la_value(norm(cuNumeric.NDArray(T[NaN, 1]))))
            @test isinf(la_value(norm(cuNumeric.NDArray(T[Inf, 1]))))
            # Unscaled powers follow cuPyNumeric's overflow/underflow tradeoff.
            @test isinf(la_value(norm(cuNumeric.NDArray(fill(T(floatmax(R)/4), 2)))))
            @test iszero(la_value(norm(cuNumeric.NDArray(fill(T(floatmin(R)), 2)))))

        end
    end
    cuNumeric.allowpromotion() do
        for T in Base.uniontypes(cuNumeric.SUPPORTED_ARRAY_TYPES)
            h = ones(T, 2)
            x = cuNumeric.NDArray(h)
            @test la_value(norm(x)) ≈ norm(h)
            @test la_value(norm(x, 0)) ≈ norm(h, 0)
        end
    end
end

# Prevent constant propagation of p from hiding branch-dependent return types.
Base.@noinline function check_norm_inference(x::cuNumeric.NDArray{T}, p::Real) where {T}
    R = typeof(float(real(zero(T))))
    @test (@inferred norm(x, p)) isa cuNumeric.CNReal{R}
end

@testset "NDArray norm inference" begin
    cuNumeric.allowscalar(false)
    cuNumeric.allowpromotion() do
        for T in Base.uniontypes(cuNumeric.SUPPORTED_ARRAY_TYPES)
            @testset "$T" begin
                x = cuNumeric.ones(T, 3)
                R = typeof(float(real(zero(T))))
                @test (@inferred norm(x)) isa cuNumeric.CNReal{R}
                for p in (0, 1, 2, 3, -1, -2, 0.5, Inf, -Inf, NaN, 2.0f0)
                    check_norm_inference(x, p)
                end
                check_norm_inference(cuNumeric.zeros(T, 0), 2)
                check_norm_inference(cuNumeric.ones(T, 2, 2), 2)
            end
        end
    end
end
