using Test
import CUDA

# One block for each possible nonempty prefix, so every warp boundary and tail
# is checked in a single launch. Invalid lanes deliberately contain nonidentity
# values to detect accidental participation in either level of the reduction.
function _test_block_reduction(src, dest, op)
    tid = Int(CUDA.threadIdx().x)
    active = Int(CUDA.blockIdx().x)
    @inbounds value = src[tid, active]
    value = cuNumeric._mr_reduce_block(op, value, active)
    tid == 1 && (@inbounds dest[active] = value)
    return nothing
end

function _check_block_prefixes(op, values::Vector{T}) where {T}
    host = fill(T(3), 256, 256)
    for active in 1:256
        host[1:active, active] .= values[1:active]
    end
    src, dest = CUDA.CuArray(host), CUDA.zeros(T, 256)
    try
        CUDA.@cuda threads=256 blocks=256 _test_block_reduction(src, dest, op)
        expected = [foldl(op, @view(values[1:n])) for n in 1:256]
        @test isequal(Array(dest), expected)
    finally
        CUDA.synchronize()
        CUDA.unsafe_free!(src)
        CUDA.unsafe_free!(dest)
    end
end

@testset "Block reduction valid prefixes" begin
    for T in (Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64,
              Float32, Float64, ComplexF32, ComplexF64)
        ops = T <: Complex ? (T === ComplexF64 ? (+,) : (+, *)) : (+, *, min, max)
        for op in ops
            values = op === (*) ? fill(one(T), 256) :
                     op === min ? fill(T(5), 256) : T[isodd(i) for i in 1:256]
            _check_block_prefixes(op, values)
        end
    end
    for T in (Float32, Float64), op in (+, *, min, max),
        value in (-zero(T), zero(T), T(NaN), T(Inf), -T(Inf))
        _check_block_prefixes(op, fill(value, 256))
    end
    for value in (ComplexF32(Inf, 0), ComplexF64(0, Inf))
        _check_block_prefixes(+, fill(value, 256))
    end
end

@testset "Runtime reduction axes" begin
    for (shape, strides) in (((5,), (2,)), ((2, 3), (2, 7)), ((2, 2, 3), (2, 7, 19)))
        N = length(shape)
        descriptor = cuNumeric.CuStridedDeviceArray{Int32,N,CUDA.AS.Global}(
            reinterpret(Core.LLVMPtr{Int32,CUDA.AS.Global}, UInt(0)),
            0, shape, strides, prod(shape),
        )
        mapper_type = typeof(cuNumeric.MapReduceMap(identity, +, Int32, ntuple(_ -> true, N)))
        for bits in 0:(2^N - 1)
            mask = ntuple(d -> !iszero(bits & (1 << (d - 1))), N)
            mapper = @inferred cuNumeric.MapReduceMap(identity, +, Int32, mask)
            @test typeof(mapper) === mapper_type
            @test mapper.mask === mask
            reduced = CartesianIndices(ntuple(d -> mask[d] ? shape[d] : 1, N))
            retained = CartesianIndices(ntuple(d -> mask[d] ? 1 : shape[d], N))
            for (r, ri) in enumerate(reduced), (o, oi) in enumerate(retained)
                expected = sum(d -> (ri[d] + oi[d] - 2) * strides[d], 1:N)
                @test (@inferred cuNumeric._mr_offset(descriptor, o - 1, r - 1, mask)) == expected
            end
        end
    end
end

function _test_strided_partial(parent, scratch, mapper, origin, stride, n, chunks)
    T = eltype(parent)
    src = cuNumeric.CuStridedDeviceArray{T,1,CUDA.AS.Global}(
        pointer(parent, origin + 1), (length(parent) - origin) * sizeof(T), (n,), (stride,), n,
    )
    dst = cuNumeric.CuStridedDeviceArray{T,1,CUDA.AS.Global}(
        pointer(scratch), length(scratch) * sizeof(T), (length(scratch),), (1,), length(scratch),
    )
    cuNumeric._mr_partial_kernel(src, dst, mapper, 0, chunks)
    return nothing
end

@testset "1D reduction offsets" begin
    mapper = cuNumeric.MapReduceMap(identity, +, Int32, (true,))
    for n in (1, 31, 256, 257, 1025, 4097), origin in (0, 3), stride in (1, 2, 3)
        host = Int32[mod(i, 7) - 3 for i in 1:(3n + 7)]
        chunks = cld(n, 1024)
        parent, scratch = CUDA.CuArray(host), CUDA.zeros(Int32, chunks)
        try
            CUDA.@cuda threads=256 blocks=chunks _test_strided_partial(
                parent, scratch, mapper, origin, stride, n, chunks,
            )
            expected = sum(host[(origin + 1):stride:(origin + 1 + (n - 1)*stride)])
            @test sum(Array(scratch)) == expected
        finally
            CUDA.synchronize()
            CUDA.unsafe_free!(parent)
            CUDA.unsafe_free!(scratch)
        end
    end
end

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
    src = cuNumeric.CuStridedDeviceArray{S,1,CUDA.AS.Global}(
        pointer(scratch), length(scratch) * sizeof(S), (length(scratch),), (1,), length(scratch),
    )
    cuNumeric._mr_contribute_full_kernel(src, dest, op, single, 0, 1, chunks)
    return nothing
end

@testset "Cooperative full contribution" begin
    for S in (Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64,
              Float32, Float64, ComplexF32, ComplexF64)
        ops = S <: Complex ? (S === ComplexF64 ? (+,) : (+, *)) : (+, *, min, max)
        for op in ops, n in (1, 31, 32, 33, 63, 65, 255, 256, 257, 511, 513,
                            1023, 1024, 1025, 4095, 4096), single in (false, true)
            host = S[isodd(i) for i in 1:n]
            seed = S(3)
            expected = foldl(op, host)
            single || (expected = op(seed, expected))
            scratch, dest = CUDA.CuArray(host), CUDA.CuArray([seed])
            try
                CUDA.@cuda threads=256 blocks=1 _test_full_contribution(scratch, dest, op, single, n)
                @test isequal(only(Array(dest)), expected)
            finally
                CUDA.synchronize()
                CUDA.unsafe_free!(scratch)
                CUDA.unsafe_free!(dest)
            end
        end
    end
end

struct ReductionSingletonOnly
    offset::Float32
end
(f::ReductionSingletonOnly)(x) = x + f.offset

@testset "Singleton preparation and asynchronous output ownership" begin
    A = cuNumeric.ones(Float32, 1, 33)
    results = Any[]
    try
        before = Set(keys(cuNumeric._MR_PTX_CACHE))
        push!(results, mapreduce(ReductionSingletonOnly(2f0), +, A; dims=1))
        added = setdiff(Set(keys(cuNumeric._MR_PTX_CACHE)), before)
        @test !isempty(added)
        @test all(key -> key[1] === cuNumeric._mr_finish_kernel, added)
        # Queue multiple freshly allocated outputs, including decode/seed
        # finishers. Their input and temporary handles may die before execution.
        for _ in 1:8
            push!(results, mapreduce(identity, min, A; dims=2))
            push!(results, @allowpromotion mapreduce(abs2, +, A; init=2.0))
            push!(results, mapreduce(identity, *, A; dims=()))
        end
        cuNumeric.destroy!(A)
        @test (@allowscalar Array(results[1])) == fill(3f0, 1, 33)
        for i in 2:3:length(results)
            @test (@allowscalar Array(results[i])) == fill(1f0, 1, 1)
            @test (@allowscalar cuNumeric.unwrap(results[i + 1])) === 35.0
            @test (@allowscalar Array(results[i + 2])) == fill(1f0, 1, 33)
        end
    finally
        cuNumeric.destroy!(A)
        foreach(cuNumeric.destroy!, results)
        cuNumeric.issue_execution_fence(; block=true)
    end
end

@testset "Full contribution special values" begin
    for value in (-0f0, -0.0, ComplexF32(Inf, 0), ComplexF64(0, Inf)),
        n in (1, 31, 256, 257, 4096)
        host = fill(value, n)
        scratch, dest = CUDA.CuArray(host), CUDA.CuArray([zero(value)])
        try
            CUDA.@cuda threads=256 blocks=1 _test_full_contribution(scratch, dest, +, true, n)
            @test isequal(only(Array(dest)), foldl(+, host))
        finally
            CUDA.synchronize()
            CUDA.unsafe_free!(scratch)
            CUDA.unsafe_free!(dest)
        end
    end
end
