@testset "transpose" begin
    A = rand(Float64, 4, 3)
    nda = cuNumeric.NDArray(A)

    ref = transpose(A)
    out = cuNumeric.transpose(nda)

    allowscalar() do
        @test cuNumeric.compare(ref, out, atol(T_OUT), rtol(T_OUT))
    end
end

@testset "eye" begin
    for T in (Float32, Float64, Int32)
        n = 5
        ref = Matrix{T}(I, n, n)
        out = cuNumeric.eye(n; T=T)
        allowscalar() do
            @test cuNumeric.compare(ref, out, atol(T_OUT), rtol(T_OUT))
        end
    end
end

@testset "trace" begin
    A = rand(Float64, 6, 6)
    nda = cuNumeric.NDArray(A)

    ref = tr(A)
    out = cuNumeric.trace(nda)

    allowscalar() do
        @test cuNumeric.compare(ref, out, atol(T_OUT), rtol(T_OUT))
    end
end

@testset "trace with offset" begin
    A = rand(Float32, 5, 5)
    nda = cuNumeric.NDArray(A)

    for k in (-2, -1, 0, 1, 2)
        ref = sum(diag(A, k))
        out = cuNumeric.trace(nda; offset=k)

        allowscalar() do
            @test cuNumeric.compare(ref, out, atol(T_OUT), rtol(T_OUT))
        end
    end
end

@testset "diag" begin
    A = rand(Int, 6, 6)
    nda = cuNumeric.NDArray(A)

    for k in (-2, 0, 3)
        ref = diag(A, k)
        out = cuNumeric.diag(nda; k=k)

        allowscalar() do
            @test cuNumeric.compare(ref, out, atol(T_OUT), rtol(T_OUT))
        end
    end
end

@testset "ravel" begin
    A = reshape(collect(1:12), 3, 4)
    nda = cuNumeric.NDArray(A)

    ref = vec(A)
    out = cuNumeric.ravel(nda)

    allowscalar() do
        @test cuNumeric.compare(ref, out, atol(T_OUT), rtol(T_OUT))
    end
end

@testset "unique" begin
    A = [1, 2, 2, 3, 4, 4, 4, 5]
    nda = cuNumeric.NDArray(A)

    ref = unique(A)
    out = cuNumeric.unique(nda)

    # Order may or may not be guaranteed — if not, compare as sets
    @test sort(Array(out)) == sort(ref)
end
