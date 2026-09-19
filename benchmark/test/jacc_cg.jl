# Run with the JACC environment; no cuNumeric or Legate imports.
using JACC, Test
JACC.@init_backend
include("../src/model_worker.jl")
include("../src/jacc/benchmarks/cg.jl")

# Exercise halo indexing on CPU, including internal partition boundaries.
struct CGTestPart{A}
    data::A
    shift::Int
end
Base.getindex(p::CGTestPart, i) = p.data[i]
Base.setindex!(p::CGTestPart, v, i) = (p.data[i]=v)
Base.length(p::CGTestPart) = length(p.data)
JACC.Multi.ghost_shift(i::Integer, p::CGTestPart) = i+p.shift
@testset "CG partition kernels" begin
    for T in (Float32, Float64), devices in (1, 2, 3, 4)
        n=12
        width=n÷devices
        p=T.(1:n)
        Ap=zeros(T, n)
        dot=zero(T)
        for d in 1:devices
            lo=(d-1)*width+1
            hi=d*width
            halo=CGTestPart(copy(p[max(1, lo - 1):min(n, hi + 1)]), Int(lo>1))
            out=CGTestPart(view(Ap, lo:hi), 0)
            for i in 1:width
                cg_matvec(i, ones(T, width), fill(T(4), width), ones(T, width), halo, out)
                dot += cg_product(i, halo, out)
            end
            for i in 1:width
                cg_direction(i, halo, out, T(2))
            end
            @test halo.data[(1 + halo.shift):(width + halo.shift)] ≈ Ap[lo:hi]+2p[lo:hi]
        end
        A=Tridiagonal(ones(T, n-1), fill(T(4), n), ones(T, n-1))
        @test Ap ≈ A*p
        @test dot ≈ sum(p .* (A*p))
    end
end
@testset "CG halo exchange" begin
    parts = [[1, 2, 3, 4, 0], [0, 5, 6, 7, 8, 0], [0, 9, 10, 11, 12]]
    cg_exchange_halos!(parts)
    @test parts == [[1, 2, 3, 4, 5], [4, 5, 6, 7, 8, 9], [8, 9, 10, 11, 12]]
end
@testset "JACC Multi CG solve" begin
    for T in (Float32, Float64), every in (1, 4, 30), limit in (1, 40)
        b=JACCCG{T}(;
            N=16JACC.Multi.ndev(), gpus=JACC.Multi.ndev(), check_every=every, max_iter=limit
        )
        @test model_check_correctness(b, nothing)=="pass"
    end
end
