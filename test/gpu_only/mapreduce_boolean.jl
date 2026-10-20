function _check_boolean_mapreduce(f, op, input; kwargs...)
    # Julia 1.10's Array reduction throws BoundsError for a zero-dimensional
    # input with dims=1. Use the equivalent singleton vector as the oracle,
    # restoring the zero-dimensional shape that NDArray preserves.
    singleton_reference = ndims(input) == 0 && get(kwargs, :dims, :) === 1
    expected = try
        reference = mapreduce(f, op, singleton_reference ? reshape(input, 1) : input; kwargs...)
        singleton_reference ? reshape(reference, ()) : reference
    catch err
        err
    end
    A = NDArray(input)
    result = nothing
    try
        if expected isa Exception
            @test_throws typeof(expected) mapreduce(f, op, A; kwargs...)
            return
        end
        result = mapreduce(f, op, A; kwargs...)
        @test size(result) == size(expected)
        T = expected isa AbstractArray ? eltype(expected) : typeof(expected)
        if result isa cuNumeric.CNScalar
            @test eltype(result.value) === T
            @test isequal(fetch(result), expected isa AbstractArray ? only(expected) : expected)
            T === Bool && (@test result isa cuNumeric.CNBool)
        else
            @test result isa NDArray{T}
            @test isequal(Array(result), expected)
        end
    finally
        isnothing(result) || cuNumeric.destroy!(result)
        cuNumeric.destroy!(A)
    end
end

struct BooleanReductionAbove{T}
    cutoff::T
end
(f::BooleanReductionAbove)(x) = x > f.cutoff

function _check_predicate_reduction(reduce, f, input; dims=:)
    # Match the zero-dimensional reference workaround above on Julia 1.10.
    singleton_reference = ndims(input) == 0 && dims === 1
    # Base's dimensional API accepts Function, while cuNumeric also supports
    # GPU-compilable callable structs.
    expected = reduce(x -> f(x), singleton_reference ? reshape(input, 1) : input; dims)
    singleton_reference && (expected = reshape(expected, ()))
    A = NDArray(input)
    result = nothing
    try
        result = try
            reduce(f, A; dims)
        catch err
            # Render errors while their NDArray arguments are still alive.
            error(sprint(showerror, err))
        end
        @test size(result) == size(expected)
        if result isa cuNumeric.CNScalar
            @test result isa cuNumeric.CNBool
            @test fetch(result) === (expected isa AbstractArray ? only(expected) : expected)
        else
            @test result isa NDArray{Bool}
            @test isequal(Array(result), expected)
        end
    finally
        isnothing(result) || cuNumeric.destroy!(result)
        cuNumeric.destroy!(A)
    end
end

@testset "Predicate any and all" begin
    for reduce in (any, all), dims in (:, 1, 2, (1, 2), (), 3),
        (f, input) in ((isfinite, Float32[1 Inf; NaN 2]),
                      (isnan, Float32[1 Inf; NaN 2]),
                      (identity, Bool[true false; true true]),
                      (!, Bool[true false; true true]),
                      (iseven, Int32[1 2; 3 4]),
                      (isfinite, ComplexF32[1 Inf; NaN 2]))
        _check_predicate_reduction(reduce, f, input; dims)
    end
    for reduce in (any, all), shape in ((0,), (0, 3), (3, 0), (2, 0, 3)),
        dims in (:, 1, 2, (1, 2), (), 4)
        _check_predicate_reduction(reduce, x -> x > 0, zeros(Int32, shape); dims)
    end
    for reduce in (any, all), value in (0f0, NaN32), shape in ((), (1,), (1, 1)),
        dims in (:, 1, ())
        _check_predicate_reduction(reduce, isfinite, fill(value, shape); dims)
    end
    for reduce in (any, all), cutoff in (0f0, 512f0, 2048f0), dims in (:, 1)
        predicate = let cutoff = cutoff
            x -> x > cutoff
        end
        _check_predicate_reduction(reduce, predicate, Float32.(1:1025); dims)
        _check_predicate_reduction(reduce, BooleanReductionAbove(cutoff), Float32.(1:1025); dims)
    end
    cuNumeric.allowfetch(false) do
        for reduce in (any, all)
            A = cuNumeric.ones(Float32, 1025)
            result = reduce(isfinite, A)
            cuNumeric.destroy!(A)
            try
                @test result isa cuNumeric.CNBool
                @test_throws Exception Bool(result)
                @test fetch(result) === true
            finally
                cuNumeric.destroy!(result)
            end
        end
    end
    for reduce in (any, all)
        A = NDArray(Int32[0, 1])
        try
            @test_throws ArgumentError reduce(identity, A)
            @test_throws ArgumentError reduce(identity, A; dims=1)
            @test_throws MethodError reduce(iszero, A; init=true)
        finally
            cuNumeric.destroy!(A)
        end
    end
end

@testset "Boolean mapped reductions" begin
    @testset "Predicates, identities, and axes" begin
        bools = reshape(Bool[isodd(i) for i in 1:105], 7, 3, 5)
        reals = reshape(Float32[mod(i, 11) == 0 ? NaN : mod(i, 7) == 0 ? Inf : i for i in 1:105], 7, 3, 5)
        integers = reshape(Int32.(1:105), 7, 3, 5)
        for (f, input) in ((identity, bools), (!, bools), (isfinite, reals),
                           (isnan, reals), (iseven, integers)),
            op in (&, |), dims in (:, 1, 2, (1, 3), (3, 1), (1, 1), (), 4)
            _check_boolean_mapreduce(f, op, input; dims)
            for init in (false, true)
                _check_boolean_mapreduce(f, op, input; dims, init)
            end
        end
        # The same closure type must use its current capture values.
        for cutoff in (0f0, 50f0, 200f0), op in (&, |)
            predicate = let cutoff = cutoff
                x -> x > cutoff
            end
            _check_boolean_mapreduce(predicate, op, reals)
            _check_boolean_mapreduce(predicate, op, reals; dims=1)
        end
        for input in (Float64[1, Inf, NaN], ComplexF32[1, Inf, NaN]), op in (&, |)
            _check_boolean_mapreduce(isfinite, op, input)
        end
    end

    @testset "Launch boundaries and truth patterns" begin
        for op in (&, |), n in (1, 31, 32, 33, 255, 256, 257, 1023, 1024, 1025, 131071)
            neutral = op === (&)
            input = fill(neutral, n)
            _check_boolean_mapreduce(identity, op, input)
            input[1] = !neutral
            _check_boolean_mapreduce(identity, op, input)
            reverse!(input)
            _check_boolean_mapreduce(identity, op, input)
        end
        for value in (false, true), shape in ((), (1,), (1, 1)),
            op in (&, |), dims in (:, 1, ())
            _check_boolean_mapreduce(identity, op, fill(value, shape); dims)
            _check_boolean_mapreduce(identity, op, fill(value, shape); dims, init=!value)
        end
    end

    @testset "Empty reduced and retained axes" begin
        for shape in ((0,), (0, 3), (3, 0), (2, 0, 3)),
            (f, T) in ((identity, Bool), (isfinite, Float32), (x -> x > 0, Int32)),
            op in (&, |), dims in (:, 1, 2, (1, 2), (), 4)
            input = zeros(T, shape)
            _check_boolean_mapreduce(f, op, input; dims)
            for init in (false, true)
                _check_boolean_mapreduce(f, op, input; dims, init)
            end
        end
    end

    @testset "Integer seeds and result types" begin
        @allowpromotion for op in (&, |), T in (Int8, UInt8, Int32, UInt64),
            init in (zero(T), one(T), T(2), typemin(T), typemax(T)),
            input in (fill(true, 7, 3), fill(false, 7, 3), zeros(Bool, 0, 3)),
            dims in (:, 1, ())
            _check_boolean_mapreduce(identity, op, input; dims, init)
        end
        cuNumeric.allowpromotion(false) do
            A = NDArray(Bool[true, false])
            try
                @test_throws Exception mapreduce(identity, &, A; init=Int32(1))
                @test_throws Exception mapreduce(identity, |, A; dims=1, init=Int32(0))
            finally
                cuNumeric.destroy!(A)
            end
        end
    end

    @testset "Sliced stores and deferred results" begin
        host = reshape(Float32.(1:323), 17, 19)
        host[8, 9] = NaN
        parent = NDArray(host)
        sliced = parent[2:16, 3:18]
        try
            for (f, op) in ((isfinite, &), (isnan, |)), dims in (:, 1, 2)
                result = mapreduce(f, op, sliced; dims)
                try
                    actual = result isa cuNumeric.CNScalar ? fetch(result) : Array(result)
                    @test isequal(actual, mapreduce(f, op, host[2:16, 3:18]; dims))
                finally
                    cuNumeric.destroy!(result)
                end
            end
        finally
            cuNumeric.destroy!(sliced)
            cuNumeric.destroy!(parent)
        end
        cuNumeric.allowfetch(false) do
            A = cuNumeric.ones(Float32, 1025)
            result = mapreduce(isfinite, &, A; init=true)
            cuNumeric.destroy!(A)
            next = nothing
            try
                @test result isa cuNumeric.CNBool
                @test_throws Exception Bool(result)
                next = mapreduce(identity, |, result)
                cuNumeric.destroy!(result)
                @test fetch(next) === true
            finally
                cuNumeric.destroy!(result)
                isnothing(next) || cuNumeric.destroy!(next)
            end
        end
    end

    @testset "Reject non-Boolean mappings" begin
        for host in (Int32[0, 1], Float32[0, 1], Int32[]), op in (&, |), dims in (:, 1)
            A = NDArray(host)
            try
                @test_throws ArgumentError mapreduce(identity, op, A; dims)
            finally
                cuNumeric.destroy!(A)
            end
        end
        A = NDArray(Bool[true, false])
        try
            @test_throws ArgumentError mapreduce(identity, &, A; init=0f0)
            @test_throws ArgumentError mapreduce(identity, |, A; dims=1, init=0f0)
        finally
            cuNumeric.destroy!(A)
        end
    end
end
