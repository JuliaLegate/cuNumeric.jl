using Test
import CUDACore

function _check_full_reduction(input, op; kwargs...)
    A = @allowscalar cuNumeric.NDArray(input)
    result = nothing
    try
        expected = mapreduce(identity, op, input; kwargs...)
        result = mapreduce(identity, op, A; kwargs...)
        actual = @allowscalar cuNumeric.unwrap(result)
        @test typeof(actual) === typeof(expected)
        @test isequal(actual, expected)
    finally
        isnothing(result) || cuNumeric.destroy!(result)
        cuNumeric.destroy!(A)
    end
end

@testset "Full reduction launch boundaries" begin
    @allowpromotion begin
        for T in (Bool, Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64,
                  Float32, Float64, ComplexF32, ComplexF64)
            ops = T <: Complex ? (T === ComplexF64 ? (+,) : (+, *)) : (+, *, min, max)
            for n in (1, 31, 32, 33, 255, 256, 257, 1023, 1024, 1025), op in ops
                _check_full_reduction(T[isodd(i) for i in 1:n], op)
            end
        end
        for T in (Float32, Float64), op in (+, *, min, max), n in (1, 257, 1025)
            for seed in (T(3), Float64(3))
                _check_full_reduction(fill(one(T), n), op; init=seed)
            end
        end
        for T in (Float32, Float64), op in (min, max)
            for x in (-zero(T), zero(T), T(NaN), T(Inf), -T(Inf))
                _check_full_reduction(fill(x, 257), op)
            end
        end
        for x in (-0f0, ComplexF32(Inf, 0), ComplexF32(0, Inf)), op in (+, *)
            _check_full_reduction([x], op)
        end
    end
    cuNumeric.issue_execution_fence(; block=true)
end

@testset "Full reduction of a sliced store" begin
    host = reshape(Int32.(1:323), 17, 19)
    parent = @allowscalar cuNumeric.NDArray(host)
    sliced = parent[2:16, 3:18]
    result = nothing
    try
        result = mapreduce(x -> x*x, +, sliced)
        cuNumeric.destroy!(sliced)
        sliced = nothing
        cuNumeric.destroy!(parent)
        parent = nothing
        @test (@allowscalar cuNumeric.unwrap(result)) == mapreduce(x -> x*x, +, host[2:16, 3:18])
    finally
        isnothing(result) || cuNumeric.destroy!(result)
        isnothing(sliced) || cuNumeric.destroy!(sliced)
        isnothing(parent) || cuNumeric.destroy!(parent)
    end
end

# Exercise the combination independently of Legate's partitioning decisions.
function _test_full_contribution(scratch, dest, op, single, chunks)
    S = eltype(scratch)
    src = cuNumeric.CuStridedDeviceArray{S,1,CUDACore.AS.Global}(
        pointer(scratch), length(scratch) * sizeof(S), (length(scratch),), (1,), length(scratch),
    )
    cuNumeric._mr_contribute_full_kernel(src, dest, op, single, 0, 1, chunks)
    return nothing
end

@testset "Cooperative full contribution" begin
    for S in (Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64,
              Float32, Float64, ComplexF32, ComplexF64)
        ops = S <: Complex ? (S === ComplexF64 ? (+,) : (+, *)) : (+, *, min, max)
        for op in ops, n in (1, 31, 32, 33, 255, 256, 257, 1024, 4096), single in (false, true)
            host = S[isodd(i) for i in 1:n]
            seed = S(3)
            expected = foldl(op, host)
            single || (expected = op(seed, expected))
            scratch, dest = CUDACore.CuArray(host), CUDACore.CuArray([seed])
            try
                CUDACore.@cuda threads=256 blocks=1 _test_full_contribution(scratch, dest, op, single, n)
                @test isequal(only(Array(dest)), expected)
            finally
                CUDACore.synchronize()
                CUDACore.unsafe_free!(scratch)
                CUDACore.unsafe_free!(dest)
            end
        end
    end
end

@testset "Full contribution special values" begin
    for value in (-0f0, -0.0, ComplexF32(Inf, 0), ComplexF64(0, Inf)),
        n in (1, 31, 256, 257, 4096)
        host = fill(value, n)
        scratch, dest = CUDACore.CuArray(host), CUDACore.CuArray([zero(value)])
        try
            CUDACore.@cuda threads=256 blocks=1 _test_full_contribution(scratch, dest, +, true, n)
            @test isequal(only(Array(dest)), foldl(+, host))
        finally
            CUDACore.synchronize()
            CUDACore.unsafe_free!(scratch)
            CUDACore.unsafe_free!(dest)
        end
    end
end
