@testset "Fusion argument planning inference" begin
    split_args = cuNumeric.split_broadcast_args_for_kernel
    runtime = cuNumeric.RuntimeBroadcastArg
    static = cuNumeric.StaticBroadcastArg

    @test (@inferred split_args(())) === ((), (), ())
    @test (@inferred split_args((1f0, 2.0))) ===
          ((1f0, 2.0), (), (runtime{1}(), runtime{2}()))
    @test (@inferred split_args((sin, Val(2)))) ===
          ((), (sin, Val(2)), (static{1}(), static{2}()))
    many_scalars = ntuple(identity, 32)
    @test (@inferred split_args(many_scalars))[1] === many_scalars

    scalar = Ref(3f0)
    args = (Ref(sin), scalar, Val(2), 4.0, Ref(Val(3)))
    expected_plan = (static{1}(), runtime{1}(), static{2}(), runtime{2}(), static{3}())
    before = @inferred split_args(args)
    @test before === ((3f0, 4.0), (sin, Val(2), Val(3)), expected_plan)
    scalar[] = 5f0
    after = @inferred split_args(args)
    @test after[1] === (5f0, 4.0)
    @test typeof(before) === typeof(after)
    @test before[2:3] === after[2:3]

    # A flattened broadcast can carry arrays directly or in Extruded wrappers.
    a = cuNumeric.NDArray(fill(1f0, 4))
    try
        extruded = Base.Broadcast.Extruded(a, (true,), (1,))
        result = @inferred split_args((a, Ref(2f0), extruded, a, Ref(sin)))
        @test result[1] === (a, 2f0, extruded, a)
        @test result[2] === (sin,)
        @test result[3] ===
              (runtime{1}(), runtime{2}(), runtime{3}(), runtime{4}(), static{1}())
        @test (@inferred cuNumeric._align_fused_runtime_args((a, 2f0))) === (a, 2f0)
        @test (@inferred cuNumeric._align_fused_runtime_args((2f0, a))) === (2f0, a)
        @test (@inferred cuNumeric._align_fused_runtime_args((a, a))) === (a, a)
        @test (@inferred cuNumeric._align_fused_runtime_args(())) === ()
        @test (@inferred cuNumeric._align_fused_runtime_args((2.0, 3.0))) === (2.0, 3.0)
        @test (@inferred cuNumeric._align_fused_runtime_args((1f0 + 2f0im,))) ===
              (1f0 + 2f0im,)
        cuNumeric.@allowpromotion begin
            @test (@inferred cuNumeric._align_fused_runtime_args((2f0, 3.0))) === (2.0, 3.0)
            @test (@inferred cuNumeric._align_fused_runtime_args((a, 2f0, 3.0))) ===
                  (a, 2.0, 3.0)
        end
    finally
        cuNumeric.destroy!(a)
    end

    # Captured data cannot be passed by the current GPU launcher.
    @test_throws ArgumentError split_args((Ref((1f0, 2f0)),))
    @test_throws ArgumentError split_args(((1f0, 2f0),))
    @test_throws ArgumentError split_args((Ref([1f0]),))
    @test_throws ArgumentError split_args(([1f0],))
end
