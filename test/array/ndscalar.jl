using Test, LinearAlgebra, cuNumeric

struct NDScalarRealSlot{T<:Real}
    value::T
end
struct NDScalarNumberSlot{T<:Number}
    value::T
end

@testset "NDScalar storage and arithmetic" begin
    autounwrap(false)
    cuNumeric.allowscalar(false)
    cuNumeric.allowpromotion() do
        for T in Base.uniontypes(cuNumeric.SUPPORTED_ARRAY_TYPES)
            a = NDArray(one(T))
            x = ndscalar(a)
            @test x.value === a
            @test x isa Number
            @test x isa supertype(T)
            @test isconcretetype(fieldtype(typeof(x), :value))
            @test x isa NDScalar{T}
            @test NDScalarNumberSlot(x).value === x
            if T <: Real
                @test x isa Real
                @test NDScalarRealSlot(x).value === x
            end
            @test unwrap(x) == one(T)
            for op in (+, -, *, /, ^)
                for (l, r) in ((x, x), (x, one(T)), (one(T), x))
                    result = @inferred op(l, r)
                    @test result isa NDScalar
                    @test unwrap(result) ≈ op(one(T), one(T))
                end
            end
            @test unwrap(zero(x)) == zero(T)
            @test unwrap(one(x)) == one(T)
            for op in (abs, abs2, sqrt, conj, real, imag, inv)
                @test unwrap(op(x)) ≈ op(one(T))
            end
            @test_throws ArgumentError convert(T, x)
            @test autounwrap(() -> convert(T, x)) == one(T)
            @test_throws ArgumentError x == one(T)
            @test (@autounwrap x == one(T))
            @test_throws ArgumentError x == one(T)
        end
        x = sum(NDArray([1.0, 2.0, 3.0]))
        @test x isa NDReal{Float64}
        @test unwrap(x^2) == 36.0
        @test unwrap(sqrt(x)) ≈ sqrt(6.0)
        @test Array(NDArray([1.0, 2.0]) .* x) == [6.0, 12.0]
        @test unwrap(max(x, 2.0)) == 6.0
        @test (@autounwrap x > 2.0)
        @test (@autounwrap isless(2.0, x))
        @test (@autounwrap 2x) isa NDScalar
        p, q = promote(x, 2.0)
        @test p isa NDScalar && q isa NDScalar
        @test unwrap(p) == 6.0 && unwrap(q) == 2.0
        for host in (3.0, 1.0 + 2.0im)
            p, q = promote(ndscalar(NDArray(2.0f0)), host)
            @test p isa NDScalar && q isa NDScalar
            @test unwrap(p) == 2.0 && unwrap(q) == host
        end
        fractional = ndscalar(NDArray(1.5))
        @test_throws ArgumentError NDInt{Int64}(fractional)
        @test_throws InexactError @autounwrap NDInt{Int64}(fractional)
        imaginary = ndscalar(NDArray(1.0 + 2.0im))
        @test_throws ArgumentError NDFloat{Float64}(imaginary)
        @test_throws InexactError @autounwrap NDFloat{Float64}(imaginary)
    end
end

@testset "Device scalar parent storage" begin
    for T in (Float64, Int64, UInt64, Bool, ComplexF64)
        host = fill(one(T))
        a = NDArray(host)
        x = @inferred ndscalar(a)
        @test x.value === a
        @test x.value.parent === host
        @test isconcretetype(fieldtype(typeof(x), :value))
        @test unwrap(x) == one(T)
    end
end

@testset "Scoped autounwrap" begin
    x = ndscalar(NDArray(2.0))
    @test autounwrap(() -> 42) == 42
    @test (@autounwrap 43) == 43
    @test_throws ErrorException autounwrap() do
        error("scope test")
    end
    @test_throws ArgumentError Float64(x)
    @autounwrap begin
        @test Float64(x) == 2.0
        autounwrap(false) do
            @test_throws ArgumentError Float64(x)
        end
        @test Float64(x) == 2.0
        # Independent tasks do not inherit this permission.
        @test fetch(@async get(task_local_storage(), :cuNumericAutoUnwrap, false)) == false
    end
    @test_throws ArgumentError Float64(x)
    @test_throws ErrorException @autounwrap error("macro scope test")
    @test_throws ArgumentError Float64(x)
    autounwrap(true)
    @test Float64(x) == 2.0
    autounwrap(false)
    @test_throws ArgumentError Float64(x)
end

@testset "Reduction return boundaries" begin
    x = NDArray([1.0, 2.0, 3.0])
    for f in (sum, prod, minimum, maximum, cuNumeric.mean, cuNumeric.var, cuNumeric.std)
        result = f(x)
        @test result isa NDReal
        @test unwrap(result) ≈ f([1.0, 2.0, 3.0])
        @test f(x; dims=1) isa NDArray{<:Any,1}
    end
    @test all(NDArray([true, true])) isa NDReal{Bool}
    @test unwrap(any(NDArray([false, true])))
    @test dot(x,x) isa NDReal
    @test unwrap(dot(x,x)) == 14.0
    z = NDArray(ComplexF64[1+2im, 3-im])
    @test dot(z,z) isa NDComplex
    @test unwrap(dot(z,z)) ≈ dot(ComplexF64[1+2im, 3-im], ComplexF64[1+2im, 3-im])
    @test (x == x) isa NDReal{Bool}
    @test unwrap(x != NDArray([3.0, 2.0, 1.0]))
    @test cuNumeric.trace(NDArray([1.0 2.0; 3.0 4.0])) isa NDReal
end
