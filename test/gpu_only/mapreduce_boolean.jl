struct BooleanReductionAbove{T}
    cutoff::T
end
(f::BooleanReductionAbove)(x) = x > f.cutoff

function check_boolean_reduction(reduce, input; reference=reduce)
    expected = reference(input)
    A, result = NDArray(input), nothing
    try
        result = reduce(A)
        T = eltype(expected)
        @test size(result) == size(expected)
        @test result isa (ndims(expected) == 0 ? cuNumeric.CNScalar : NDArray{T})
        actual = result isa cuNumeric.CNScalar ? fetch(result) : Array(result)
        @test eltype(actual) === T
        @test isequal(actual, expected isa AbstractArray && ndims(expected) == 0 ? only(expected) : expected)
    catch err
        # Format exceptions before destroying any NDArray arguments they contain.
        error(sprint(showerror, err))
    finally
        isnothing(result) || cuNumeric.destroy!(result)
        cuNumeric.destroy!(A)
    end
end

@testset "Boolean mapped reductions and predicates" begin
    inputs = (Float32[1 Inf; NaN 2], zeros(Float32, 0, 3), zeros(Float32, 3, 0))
    for (op, reduce, neutral) in ((&, all, true), (|, any, false)),
        f in (isfinite, isnan), input in inputs, dims in (:, 1, 2, (), (1, 2))
        check_boolean_reduction(A -> reduce(f, A; dims), input)
        check_boolean_reduction(A -> mapreduce(f, op, A; dims, init=neutral), input)
    end
    for op in (&, |), input in (Bool[], Bool[true, false], fill(true)), dims in (:, ())
        check_boolean_reduction(A -> mapreduce(identity, op, A; dims), input)
    end
    for op in (&, |), neutral in (false, true)
        input = fill(neutral, 1025)
        check_boolean_reduction(A -> mapreduce(identity, op, A), input)
        input[end] = !neutral
        check_boolean_reduction(A -> mapreduce(identity, op, A), input)
        check_boolean_reduction(A -> mapreduce(identity, op, A; init=!neutral), input)
    end
    # Changing captured values must work with a cached predicate specialization.
    for cutoff in (0f0, 5f0), reduce in (any, all)
        predicate = let cutoff = cutoff
            x -> x > cutoff
        end
        check_boolean_reduction(A -> reduce(predicate, A), Float32[1, 2])
        functor = BooleanReductionAbove(cutoff)
        check_boolean_reduction(A -> reduce(functor, A; dims=1), Float32[1, 2];
            reference=A -> reduce(x -> functor(x), A; dims=1))
    end
    # Integer seeds affect result typing, including signed, non-neutral seeds.
    @allowpromotion for op in (&, |), init in (Int32(2), typemin(Int32)), dims in (:, 1)
        check_boolean_reduction(A -> mapreduce(identity, op, A; dims, init), Bool[true, false])
    end
    A = NDArray(Float32[1, NaN])
    try
        for reduce in (any, all)
            result = reduce(isfinite, A)
            try
                @test result isa cuNumeric.CNBool
                cuNumeric.allowfetch(false) do
                    @test_throws Exception Bool(result)
                end
                @test fetch(result) === reduce(isfinite, Float32[1, NaN])
            finally
                cuNumeric.destroy!(result)
            end
            @test_throws ArgumentError reduce(identity, A)
            @test_throws MethodError reduce(isfinite, A; init=true)
        end
        for op in (&, |)
            @test_throws ArgumentError mapreduce(identity, op, A)
            @test_throws ArgumentError mapreduce(isfinite, op, A; dims=1, init=0f0)
        end
    finally
        cuNumeric.destroy!(A)
    end
end
