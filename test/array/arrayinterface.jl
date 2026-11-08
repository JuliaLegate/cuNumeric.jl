using Test, cuNumeric, LinearAlgebra
import ArrayInterface as AI

@testset "ArrayInterface extension" begin
    allowscalar(false) do
        host = reshape(Float32.(1:6), 2, 3)
        A = NDArray(host)
        for x in (A, view(A, :, 1), transpose(A), adjoint(A))
            @test !AI.fast_scalar_indexing(typeof(x))
        end
        @test AI.allowed_getindex(A, 2, 3) == 6.0f0
        AI.allowed_setindex!(A, 10.0f0, 1, 2)
        @test Array(A)[1, 2] == 10.0f0
        @test_throws ErrorException A[1, 1]

        # Check Julia element order and source eltype for host and backend inputs.
        host = reshape(Float64.(1:6), 2, 3)
        for source in (host, NDArray(host)), dims in ((6,), (3, 2))
            template = cuNumeric.zeros(Float32, dims)
            result = @inferred AI.restructure(template, source)
            @test result isa NDArray{Float64,length(dims)}
            @test Array(result) == reshape(host, dims)
        end

        result = @inferred AI.zeromatrix(A)
        @test result isa NDArray{Float32,2}
        @test Array(result) == zeros(Float32, length(A), length(A))
    end
end
