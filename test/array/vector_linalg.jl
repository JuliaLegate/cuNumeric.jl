using Test, LinearAlgebra, Random, cuNumeric

# Host extraction belongs to validation, not the NDArray implementation.
la_value(x::cuNumeric.CNScalar) = fetch(x)

@testset "LinearAlgebra vector interface" begin
    cuNumeric.allowscalar(false)
    begin
        Random.seed!(912)
        for T in Base.uniontypes(
            Union{cuNumeric.SUPPORTED_FLOAT_TYPES,cuNumeric.SUPPORTED_COMPLEX_TYPES}
        )
            @testset "$T" begin
                R = real(T)
                ah, xh, yh = randn(T, 7, 5), randn(T, 5), randn(T, 7)
                a, x, y = cuNumeric.NDArray(ah), cuNumeric.NDArray(xh), cuNumeric.NDArray(yh)
                @test mul!(y, a, x) === y
                @test Array(y) ≈ ah * xh
                @test Array(a * x) ≈ ah * xh
                copyto!(y, cuNumeric.NDArray(yh))
                @test mul!(y, a, x, T(2), T(-3)) === y
                @test Array(y) ≈ 2ah * xh - 3yh
                fill!(y, T(NaN))
                mul!(y, a, x, T(2), zero(T))
                @test Array(y) ≈ 2ah * xh
                bad = cuNumeric.NDArray(fill(T(NaN), 7, 5))
                mul!(y, bad, x, zero(T), zero(T))
                @test all(iszero, Array(y))
                @test_throws DimensionMismatch mul!(similar(x), a, x)
                square = cuNumeric.NDArray(randn(T, 5, 5))
                @test_throws ArgumentError mul!(x, square, x)
                bh = randn(T, 5, 3)
                ch = randn(T, 7, 3)
                c = cuNumeric.NDArray(ch)
                mul!(c, a, cuNumeric.NDArray(bh), T(2), T(3))
                @test Array(c) ≈ 2ah * bh + 3ch

                zh = randn(T, 5)
                z = cuNumeric.NDArray(zh)
                @test dot(x, z) isa (T <: Real ? CNReal{T} : CNComplex{T})
                @test la_value(dot(x, z)) ≈ dot(xh, zh)
                @test_throws DimensionMismatch dot(x, y)
                empty = cuNumeric.zeros(T, 0)
                @test la_value(dot(empty, empty)) === zero(T)
                @test axpy!(T(2), x, z) === z
                @test Array(z) ≈ 2xh + zh
                @test axpby!(T(-2), x, T(3), z) === z
                @test Array(z) ≈ -2xh + 3(2xh + zh)
                @test axpy!(one(T), x, x) === x
                @test Array(x) ≈ 2xh
                @test axpby!(T(2), x, T(3), x) === x
                @test Array(x) ≈ 10xh
                @test rmul!(x, T(2)) === x
                @test lmul!(T(3), x) === x
                @test Array(x) ≈ 60xh
                @test_throws DimensionMismatch axpy!(one(T), x, y)
                @test_throws DimensionMismatch axpby!(one(T), x, one(T), y)
                d = Diagonal(cuNumeric.NDArray(T[1, 2, 3, 4, 5]))
                ldiv!(z, d, x)
                @test Array(z) ≈ 60xh ./ T[1, 2, 3, 4, 5]

                # Vector views must stay on the backend for updates and products.
                storage = cuNumeric.zeros(T, 9)
                dest = view(storage, 2:8)
                mul!(dest, a, cuNumeric.NDArray(xh))
                @test Array(storage)[2:8] ≈ ah * xh
            end
        end
        empty_a = cuNumeric.zeros(Float64, 3, 0)
        out = cuNumeric.ones(Float64, 3)
        mul!(out, empty_a, cuNumeric.zeros(Float64, 0))
        @test all(iszero, Array(out))
        cuNumeric.allowpromotion() do
            a32 = cuNumeric.NDArray(Float32[2 1; 0 3])
            x64 = cuNumeric.NDArray([2.0, 4.0])
            y64 = similar(x64)
            mul!(y64, a32, x64)
            @test Array(y64) ≈ [8.0, 12.0]
            @test la_value(dot(cuNumeric.NDArray(Float32[1, 2]), x64)) ≈ 10.0
            @test_throws ArgumentError mul!(cuNumeric.zeros(Float32, 2), a32, x64)
        end
    end
end

@testset "LinearAlgebra promotion across numeric types" begin
    cuNumeric.allowscalar(false)
    cuNumeric.allowpromotion() do
        types = Base.uniontypes(cuNumeric.SUPPORTED_ARRAY_TYPES)
        for T in types
            @testset "updates $T" begin
                h = ones(T, 2)
                y = cuNumeric.ones(T,2)
                z = cuNumeric.zeros(T,2)
                @test axpy!(one(T),z,y) === y
                @test Array(y) == h
                @test axpby!(one(T),z,one(T),y) === y
                @test Array(y) == h
                @test lmul!(one(T),y) === y
                @test rmul!(y,one(T)) === y
                @test Array(y) == h
            end
        end
        for T in types, S in types
            @testset "$T / $S" begin
                a = cuNumeric.ones(T,2,2)
                b = cuNumeric.ones(S,2,2)
                x, z = cuNumeric.ones(S,2), cuNumeric.ones(T,2)
                expected_dot = dot(ones(T,2),ones(S,2))
                @test la_value(dot(z,x)) == expected_dot
                # Only matrix kernels reject integer-integer inputs.
                if !(T <: Integer && S <: Integer)
                    R = promote_type(T,S)
                    @test Array(a*x) ≈ fill(R(2),2)
                    y = cuNumeric.zeros(R,2)
                    @test mul!(y,a,x) === y
                    @test Array(y) ≈ fill(R(2),2)
                    mul!(y,a,x,R(2),R(3))
                    @test Array(y) ≈ fill(R(10),2)
                    c = cuNumeric.zeros(R,2,2)
                    mul!(c,a,b,R(2),R(0))
                    @test Array(c) ≈ fill(R(4),2,2)
                end
            end
        end
    end
end

@testset "Integer-integer matrix multiplication errors" begin
    cuNumeric.allowscalar(false)
    ints = Base.uniontypes(Union{Bool,cuNumeric.SUPPORTED_INT_TYPES})
    for T in ints, S in ints
        A, x = cuNumeric.ones(T, 2, 2), cuNumeric.ones(S, 2)
        B = cuNumeric.ones(S, 2, 2)
        y, C = cuNumeric.zeros(Float64, 2), cuNumeric.zeros(Float64, 2, 2)
        for call in (() -> A*x, () -> mul!(y,A,x), () -> mul!(y,A,x,1,0),
                     () -> mul!(C,A,B,1,0), () -> mul!(y,A,x,0,0))
            err = try
                call()
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            if err isa ArgumentError
                @test occursin("integer-integer", sprint(showerror,err))
                @test occursin("Convert an operand", sprint(showerror,err))
            end
        end
    end
    A = cuNumeric.ones(Float64,2,2)
    x = cuNumeric.ones(Int32,2)
    @test_throws ArgumentError mul!(cuNumeric.zeros(Int32,2), A, x)
    @test_throws DimensionMismatch A * cuNumeric.ones(Float64,3)
    cuNumeric.allowpromotion(false) do
        @test_throws "Implicit promotion" A * cuNumeric.ones(Int8,2)
    end
end

@testset "0D coefficient arithmetic" begin
    cuNumeric.allowscalar(false)
    cuNumeric.allowpromotion() do
        types = Base.uniontypes(cuNumeric.SUPPORTED_ARRAY_TYPES)
        pairs = [(T, T) for T in types]
        append!(pairs, [(Float32, Float64), (Int32, Float32), (Float64, Int8),
                        (ComplexF32, Float32), (ComplexF64, Int64), (Bool, Int8), (Int8, UInt8)])
        for (T, S) in pairs
            a, b = NDArray(one(T)), NDArray(one(S))
            for op in (+, -, *, /, ^)
                for (x, y) in ((a, b), (a, one(S)), (one(T), b))
                    result = @inferred op(x, y)
                    expected = op(one(T), one(S))
                    @test result isa NDArray{typeof(expected),0}
                    @test only(result) ≈ expected
                end
            end
        end
        for T in Base.uniontypes(Union{cuNumeric.SUPPORTED_FLOAT_TYPES,cuNumeric.SUPPORTED_COMPLEX_TYPES})
            a, b = NDArray(T(6)), NDArray(T(2))
            @test only(a-b) ≈ T(4)
            @test only(a/b) ≈ T(3)
            @test only(T(12)/a) ≈ T(2)
            @test only(T(12)-a) ≈ T(6)
            @test only(a^2) ≈ T(36)
            exponent = 3
            @test only(a^exponent) ≈ T(216)
            @test only(a^b) ≈ T(36)
            @test only(T(2)^b) ≈ T(4)
            @test only(a^(-1)) ≈ inv(T(6))
        end
    end
    # Higher-dimensional products still mean matrix multiplication.
    A = NDArray([1.0 2.0; 3.0 4.0])
    @test Array(A*A) ≈ [7.0 10.0; 15.0 22.0]
end
