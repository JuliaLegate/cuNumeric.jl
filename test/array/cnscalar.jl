using Test, LinearAlgebra, cuNumeric

struct CNScalarRealSlot{T<:Real}
    value::T
end
struct CNScalarNumberSlot{T<:Number}
    value::T
end

@testset "CNScalar storage and arithmetic" begin
    allowautofetch(false)
    cuNumeric.allowscalar(false)
    cuNumeric.allowpromotion() do
        for T in Base.uniontypes(cuNumeric.SUPPORTED_ARRAY_TYPES)
            a = NDArray(one(T))
            x = cnscalar(a)
            @test x.value === a
            @test x isa Number
            @test x isa supertype(T)
            @test isconcretetype(fieldtype(typeof(x), :value))
            @test x isa CNScalar{T}
            @test CNScalarNumberSlot(x).value === x
            if T <: Real
                @test x isa Real
                @test CNScalarRealSlot(x).value === x
            end
            @test fetch(x) == one(T)
            @test fetch(a) == one(T)
            @test only(x) == fetch(x)
            @test_throws ErrorException x[]
            @test_throws ErrorException (@allowautofetch x[])
            cuNumeric.allowscalar() do
                @test x[] === a[]
                @test x[] == one(T)
                @test_throws ArgumentError x == one(T)
            end
            for op in (+, -, *, /, ^)
                for (l, r) in ((x, x), (x, one(T)), (one(T), x))
                    result = @inferred op(l, r)
                    @test result isa CNScalar
                    @test fetch(result) ≈ op(one(T), one(T))
                end
            end
            @test fetch(zero(x)) == zero(T)
            @test fetch(one(x)) == one(T)
            for op in (abs, abs2, sqrt, conj, real, imag, inv)
                @test fetch(op(x)) ≈ op(one(T))
            end
            @test_throws ArgumentError convert(T, x)
            @test allowautofetch(() -> convert(T, x)) == one(T)
            @test_throws ArgumentError x == one(T)
            @test (@allowautofetch x == one(T))
            @test_throws ArgumentError x == one(T)
        end
        x = sum(NDArray([1.0, 2.0, 3.0]))
        @test x isa CNReal{Float64}
        @test fetch(x^2) == 36.0
        @test fetch(sqrt(x)) ≈ sqrt(6.0)
        @test Array(NDArray([1.0, 2.0]) .* x) == [6.0, 12.0]
        @test fetch(max(x, 2.0)) == 6.0
        @test (@allowautofetch x > 2.0)
        @test (@allowautofetch isless(2.0, x))
        @test (@allowautofetch 2x) isa CNScalar
        p, q = promote(x, 2.0)
        @test p isa CNScalar && q isa CNScalar
        @test fetch(p) == 6.0 && fetch(q) == 2.0
        for host in (3.0, 1.0 + 2.0im)
            p, q = promote(cnscalar(NDArray(2.0f0)), host)
            @test p isa CNScalar && q isa CNScalar
            @test fetch(p) == 2.0 && fetch(q) == host
        end
        fractional = cnscalar(NDArray(1.5))
        @test_throws ArgumentError CNInt{Int64}(fractional)
        @test_throws InexactError @allowautofetch CNInt{Int64}(fractional)
        imaginary = cnscalar(NDArray(1.0 + 2.0im))
        @test_throws ArgumentError CNFloat{Float64}(imaginary)
        @test_throws InexactError @allowautofetch CNFloat{Float64}(imaginary)
    end
end

@testset "Device scalar parent storage" begin
    for T in (Float64, Int64, UInt64, Bool, ComplexF64)
        host = fill(one(T))
        a = NDArray(host)
        x = @inferred cnscalar(a)
        @test x.value === a
        @test x.value.parent === host
        @test isconcretetype(fieldtype(typeof(x), :value))
        @test fetch(x) == one(T)
    end
end

@testset "Scoped allowautofetch" begin
    x = cnscalar(NDArray(2.0))
    @test allowautofetch(() -> 42) == 42
    @test (@allowautofetch 43) == 43
    @test_throws ErrorException allowautofetch() do
        error("scope test")
    end
    @test_throws ArgumentError Float64(x)
    @allowautofetch begin
        @test Float64(x) == 2.0
        allowautofetch(false) do
            @test_throws ArgumentError Float64(x)
        end
        @test Float64(x) == 2.0
        # Independent tasks do not inherit this permission.
        @test fetch(@async get(task_local_storage(), :cuNumericAllowAutoFetch, false)) == false
    end
    @test_throws ArgumentError Float64(x)
    @test_throws ErrorException @allowautofetch error("macro scope test")
    @test_throws ArgumentError Float64(x)
    allowautofetch(true)
    @test Float64(x) == 2.0
    allowautofetch(false)
    @test_throws ArgumentError Float64(x)
end

@testset "Reduction return boundaries" begin
    x = NDArray([1.0, 2.0, 3.0])
    for f in (sum, prod, minimum, maximum, cuNumeric.mean, cuNumeric.var, cuNumeric.std)
        result = f(x)
        @test result isa CNReal
        @test fetch(result) ≈ f([1.0, 2.0, 3.0])
        @test f(x; dims=1) isa NDArray{<:Any,1}
    end
    @test all(NDArray([true, true])) isa CNReal{Bool}
    @test fetch(any(NDArray([false, true])))
    @test dot(x,x) isa CNReal
    @test fetch(dot(x,x)) == 14.0
    z = NDArray(ComplexF64[1+2im, 3-im])
    @test dot(z,z) isa CNComplex
    @test fetch(dot(z,z)) ≈ dot(ComplexF64[1+2im, 3-im], ComplexF64[1+2im, 3-im])
    @test (x == x) isa CNReal{Bool}
    @test fetch(x != NDArray([3.0, 2.0, 1.0]))
    @test cuNumeric.trace(NDArray([1.0 2.0; 3.0 4.0])) isa CNReal
end
