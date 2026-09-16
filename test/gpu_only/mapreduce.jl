_mapped_reduction_host(A) = @allowscalar ndims(A) == 0 ? cuNumeric.unwrap(A) : Array(A)

function _mapreduce_checkpoint(stage; details...)
    get(ENV, "CUNUMERIC_MAPREDUCE_TRACE", "0") == "1" || return
    @info "mapreduce checkpoint" stage details...
    flush(stderr)
end

struct ReductionAffine
    scale::Float32
    offset::Float64
    sign::Int8
end
(f::ReductionAffine)(x) = f.scale * x + f.offset + f.sign

function _mapped_reduction_tolerances(f, input; dims=:)
    T = Base.promote_op(f, eltype(input))
    region = dims isa Integer ? (dims,) : dims
    n = prod((size(input, d) for d in 1:ndims(input) if dims isa Colon || d in region); init=1)
    # Use mapped magnitudes so cancellation and type-changing maps are covered.
    scale = maximum(x -> abs(f(x)), input; init=zero(real(T)))
    return (; rtol=reduction_rtol(T, max(n, 1)), atol=reduction_atol(T, max(n, 1), scale))
end

function _check_mapped_reduction(f, op, input; kwargs...)
    _mapreduce_checkpoint("construct input"; f, op, type=eltype(input), shape=size(input), kwargs...)
    A = @allowscalar NDArray(input)
    result = nothing
    try
        _mapreduce_checkpoint("Base reference")
        expected = mapreduce(f, op, input; kwargs...)
        _mapreduce_checkpoint("submit reduction")
        result = mapreduce(f, op, A; kwargs...)
        _mapreduce_checkpoint("submission returned")
        if get(ENV, "CUNUMERIC_MAPREDUCE_SYNC", "0") == "1"
            _mapreduce_checkpoint("execution fence")
            cuNumeric.issue_execution_fence(; block=true)
            _mapreduce_checkpoint("execution fence returned")
        end
        @test size(result) == size(expected)
        @test eltype(result) === (expected isa AbstractArray ? eltype(expected) : typeof(expected))
        _mapreduce_checkpoint("read result")
        actual = _mapped_reduction_host(result)
        _mapreduce_checkpoint("result read")
        if op === min || op === max || eltype(result) <: Integer
            @test isequal(actual, expected)
        else
            tolerances = _mapped_reduction_tolerances(f, input; dims=get(kwargs, :dims, :))
            @test isapprox(actual, expected; rtol=tolerances.rtol, atol=tolerances.atol)
        end
    finally
        _mapreduce_checkpoint("destroy input")
        cuNumeric.destroy!(A)
        _mapreduce_checkpoint("destroy result")
        isnothing(result) || cuNumeric.destroy!(result)
        _mapreduce_checkpoint("case complete")
    end
end

@testset "Fused mapped reductions" begin
    @allowpromotion begin
        for T in (Bool, Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, Float32, Float64)
            # A typed comprehension keeps Bool inputs byte-addressable, unlike broadcast.
            input = reshape(T[mod(i, 2) for i in 0:104], 7, 3, 5)
            for op in (+, *, min, max), dims in (:, 1, 2, (1, 3), (3, 1), (1, 1), (), 4)
                _check_mapped_reduction(identity, op, input; dims)
            end
        end
        for T in (ComplexF32, ComplexF64), op in (+, *)
            if T === ComplexF64 && op === (*)
                A = cuNumeric.ones(T, 7, 9)
                try
                    @test_throws ArgumentError mapreduce(identity, op, A; dims=1)
                    @test_throws ArgumentError prod(identity, A)
                finally
                    cuNumeric.destroy!(A)
                end
                continue
            end
            _check_mapped_reduction(identity, op, fill(T(1 + 0im), 7, 9); dims=1)
            _check_mapped_reduction(abs2, +, fill(T(1 + 2im), 7, 9))
        end
        for shape in ((), (1,), (1, 1)), op in (+, *, min, max)
            _check_mapped_reduction(abs2, op, fill(2f0, shape))
        end
        # Full singletons use reduce_first; dimensional sums/products seed
        # their result with zero/one. Check exact zeros and complex infinities.
        for x in (-0f0, ComplexF32(Inf, 0), ComplexF32(0, Inf)),
            op in (+, *), dims in (:, 1, ())
            input = fill(x, 1)
            A = @allowscalar NDArray(input)
            r = nothing
            try
                r = mapreduce(identity, op, A; dims)
                @test isequal(_mapped_reduction_host(r), mapreduce(identity, op, input; dims))
            finally
                cuNumeric.destroy!(A)
                isnothing(r) || cuNumeric.destroy!(r)
            end
        end
        for T in (Float32, Float64), op in (min, max)
            for pair in ((-zero(T), zero(T)), (T(NaN), T(2)), (T(-Inf), T(Inf)))
                input = fill(pair[2], 131071)
                input[1] = pair[1]
                _check_mapped_reduction(identity, op, input)
                reverse!(input)
                _check_mapped_reduction(identity, op, input)
            end
        end

        input = reshape(Float32.(1:17017) ./ 17017f0, 7, 11, 221)
        _check_mapped_reduction(ReductionAffine(2f0, 0.5, Int8(-1)), +, input)
        _check_mapped_reduction(Float64, +, input)
        _check_mapped_reduction(identity, +, Int8[100, 100])
        for init in (0f0, 0.0), dims in (:, 1, (1, 3))
            _check_mapped_reduction(abs2, +, input; init, dims)
        end
        # Non-neutral seeds expose accidental replication across partitions.
        _check_mapped_reduction(identity, +, ones(Float32, 131071); init=7f0)
        _check_mapped_reduction(identity, *, ones(Float32, 131071); init=3f0)
        _check_mapped_reduction(identity, max, ones(Float32, 131071); init=5f0)
        _check_mapped_reduction(identity, min, ones(Float32, 131071); init=-5f0)

        for dims in (:, 1), f in (identity, abs, abs2)
            _check_mapped_reduction(f, +, Float32[]; dims)
            _check_mapped_reduction(f, *, Float32[]; dims)
        end
        _check_mapped_reduction(abs2, max, Float32[])
        _check_mapped_reduction(x -> x*x, +, Float32[]; init=0f0)
        _check_mapped_reduction(x -> x*x, +, zeros(Float32, 0, 3); dims=1)
        _check_mapped_reduction(identity, min, zeros(Float32, 3, 0); dims=1)

        A = @allowscalar NDArray(Int8[2, 3, 4])
        try
            for (fn, op) in ((sum, Base.add_sum), (prod, Base.mul_prod), (minimum, min), (maximum, max))
                r = fn(identity, A)
                @test _mapped_reduction_host(r) === fn(identity, Int8[2, 3, 4])
                cuNumeric.destroy!(r)
            end
        finally
            cuNumeric.destroy!(A)
        end

        # Same closure type, different capture values must reuse PTX correctly.
        A = @allowscalar NDArray(input)
        try
            cache_size = nothing
            for alpha in (1f0, 3f0, -2f0)
                f = let alpha = alpha
                    x -> abs2(x - alpha)
                end
                r = mapreduce(f, +, A)
                tolerances = _mapped_reduction_tolerances(f, input)
                @test isapprox(
                    _mapped_reduction_host(r), mapreduce(f, +, input);
                    rtol=tolerances.rtol, atol=tolerances.atol,
                )
                cuNumeric.destroy!(r)
                current_size = length(cuNumeric._MR_PTX_CACHE)
                isnothing(cache_size) || (@test current_size == cache_size)
                cache_size = current_size
            end
            @test_throws ArgumentError mapreduce(identity, -, A)
            @test_throws ArgumentError mapreduce(+, +, A, A)
            @test_throws ArgumentError mapreduce(identity, +, A; dims=0)
            @test_throws ArgumentError mapreduce(identity, +, A; dims=1, init=Int8(0))
        finally
            cuNumeric.destroy!(A)
        end

        empty = cuNumeric.zeros(Float32, 0)
        try
            @test_throws ArgumentError mapreduce(x -> x*x, +, empty)
            @test_throws ArgumentError minimum(identity, empty)
            @test_throws ArgumentError maximum(identity, empty; dims=1)
        finally
            cuNumeric.destroy!(empty)
        end

        # A sliced logical store can have a nonzero origin and non-dense strides.
        parent = @allowscalar NDArray(reshape(Float32.(1:323), 17, 19))
        sliced = parent[2:16, 3:18]
        r = mapreduce(abs2, +, sliced; dims=1)
        @test _mapped_reduction_host(r) ≈ mapreduce(abs2, +, reshape(Float32.(1:323), 17, 19)[2:16, 3:18]; dims=1)
        cuNumeric.destroy!(r)
        cuNumeric.destroy!(sliced)
        cuNumeric.destroy!(parent)

        # Drop the source before execution completes, then consume the result
        # on the device without an intervening unwrap or execution fence.
        A = cuNumeric.ones(Float32, 131071)
        r = mapreduce(abs2, +, A)
        cuNumeric.destroy!(A)
        next = r + r
        cuNumeric.destroy!(r)
        @test _mapped_reduction_host(next) == 2f0 * 131071
        cuNumeric.destroy!(next)
    end
    cuNumeric.allowpromotion(false) do
        A = cuNumeric.ones(Int8, 3)
        @test_throws Exception sum(identity, A)
        r = mapreduce(identity, +, A)
        @test eltype(r) === Int8
        cuNumeric.destroy!(r)
        cuNumeric.destroy!(A)
    end
end
