struct ReductionPointerNumber <: Number
    ptr::Ptr{Float32}
end
(f::ReductionPointerNumber)(x) = x

@testset "Mapped reduction policies" begin
    CN = cuNumeric
    noinit = CN.NoReductionInit()
    @test CN._mr_dims((2, 3), :) == ((true, true), ())
    @test CN._mr_dims((2, 3), (1, 1, 4)) == ((true, false), (1, 3))
    @test CN._mr_dims((2, 3), ()) == ((false, false), (2, 3))
    @test_throws ArgumentError CN._mr_dims((2, 3), 0)
    @test_throws ArgumentError CN._mr_dims((2, 3), (1, 1.5))
    @test_throws ArgumentError CN._mr_operator(-)
    @test CN._mr_redop(+, Float32) isa CN.MapReduceOp
    @test CN._mr_redop(+, Float32) == CN.MAPREDUCE_ADD
    @test CN._mr_redop(Base.add_sum, Int64) == CN.MAPREDUCE_ADD
    @test CN._mr_redop(*, Float32) == CN.MAPREDUCE_MUL
    @test CN._mr_redop(Base.mul_prod, Int64) == CN.MAPREDUCE_MUL
    @test CN._mr_redop(min, UInt32) == CN.MAPREDUCE_MIN
    @test CN._mr_redop(max, UInt64) == CN.MAPREDUCE_MAX
    @test CN._mr_redop(*, Bool) == CN.MAPREDUCE_AND
    @test CN._mr_redop(min, Bool) == CN.MAPREDUCE_AND
    @test CN._mr_redop(max, Bool) == CN.MAPREDUCE_OR
    @test_throws ArgumentError CN._mr_accumulator(min, ComplexF32)
    @test_throws ArgumentError CN._mr_accumulator(+, Float64, 0f0, 1)
    @test_throws ArgumentError CN._mr_accumulator(min, Float32, ComplexF32(0), 1)
    @test_throws ArgumentError CN._mr_mapped_type(ReductionPointerNumber(Ptr{Float32}(0)), Float32)

    for T in (Bool, Int8, UInt16, Int32, UInt64, Float32, Float64, ComplexF32),
        op in (+, *, Base.add_sum, Base.mul_prod)
        @test CN._mr_accumulator(op, T) === typeof(Base.reduce_first(op, one(T)))
    end
    @testset "Empty reduction f=$f op=$op dims=$dims" for
        f in (identity, abs, abs2, x -> x*x), op in (+, *, min, max), dims in (:, 1, (1,))
        reference = try
            mapreduce(f, op, Float32[]; dims)
        catch e
            e
        end
        if reference isa Exception
            @test_throws typeof(reference) CN._mr_empty(f, op, Float32, Float32, noinit, dims)
        else
            scalar = reference isa AbstractArray ? only(reference) : reference
            @test isequal(CN._mr_empty(f, op, Float32, Float32, noinit, dims), scalar)
        end
    end
    @test CN._mr_output_type(+, Float32, 0.0, :) === Float64
    @test CN._mr_output_type(+, Float32, noinit, 1) === Float32

    # Verify ordering against Julia, including NaNs and signed zeros.
    for T in (Float32, Float64), op in (min, max)
        xs = T[-Inf, -2, -0.0, 0.0, 2, Inf, NaN]
        for a in xs, b in xs
            encoded = op(CN._mr_encode(op, a), CN._mr_encode(op, b))
            @test isequal(CN._mr_decode(op, T, encoded), op(a, b))
        end
    end
    captured = let a = [1.0f0]
        x -> x + a[1]
    end
    @test_throws ArgumentError CN._mr_mapped_type(captured, Float32)
    @test CN._mr_mapped_type(CN._mr_callable(Float64), Float32) === Float64

    if !CN._has_gpu_target()
        A = CN.ones(Float32, 4)
        @test_throws ArgumentError mapreduce(identity, +, A)
        @test_throws ArgumentError sum(abs2, A)
        CN.destroy!(A)
    end
end
