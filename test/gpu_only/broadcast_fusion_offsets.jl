_offset_test_op(x, s) = x * s + one(x)
_offset_test_pair(x, y) = x * 2 + y
_offset_unrelated_op(x) = x - 3.0f0
_offset_ordinary_call(a) = _offset_unrelated_op.(a)
struct OffsetTestRecord
    x::Float32
end
_offset_record_op(x) = OffsetTestRecord(2x.x)

@testset "Adaptive broadcast plan caching" begin
    lock(cuNumeric._BROADCAST_CACHE_LOCK) do
        empty!(cuNumeric._BROADCAST_PLAN_CACHE)
        empty!(cuNumeric._BROADCAST_VARIANT_CACHE)
    end
    wide = cuNumeric.FixedOffsets{Int64}()
    adaptive = cuNumeric.AdaptiveOffsets()
    bc(a, s) = Base.broadcasted(_offset_test_op, a, s)
    a = NDArray(reshape(Float32.(1:12), 3, 4))
    cuNumeric._materialize_with_policy(wide, bc(a, 2.0f0))
    fixed = only(values(cuNumeric._BROADCAST_PLAN_CACHE))
    @test fixed isa cuNumeric.FixedBroadcastPlan
    @test length(cuNumeric._BROADCAST_VARIANT_CACHE) == 1
    cuNumeric._materialize_with_policy(adaptive, bc(a, 2.0f0))
    pair = only(
        filter(p -> p isa cuNumeric.AdaptiveBroadcastPlan,
            collect(values(cuNumeric._BROADCAST_PLAN_CACHE))),
    )
    @test pair.wide === fixed.kernel
    @test length(cuNumeric._BROADCAST_PLAN_CACHE) == 2
    @test length(cuNumeric._BROADCAST_VARIANT_CACHE) == 2
    # New dimensions and scalar values reuse the same complete plan.
    b = NDArray(reshape(Float32.(1:30), 5, 6))
    @test Array(cuNumeric._materialize_with_policy(adaptive, bc(b, 3.0f0))) == Array(b) .* 3 .+ 1
    @test length(cuNumeric._BROADCAST_PLAN_CACHE) == 2
    @test length(cuNumeric._BROADCAST_VARIANT_CACHE) == 2
    cuNumeric._materialize_with_policy(adaptive, bc(b, 4.0f0))
    @test length(cuNumeric._BROADCAST_VARIANT_CACHE) == 2
    slice = b[1:3, 2:5]
    result = cuNumeric._materialize_with_policy(adaptive, bc(slice, 6.0f0))
    @test Array(result) == Array(b)[1:3, 2:5] .* 6 .+ 1
    @test length(cuNumeric._BROADCAST_PLAN_CACHE) == 2
    @test length(cuNumeric._BROADCAST_VARIANT_CACHE) == 2
    # The macro must not transmit its policy into an ordinary callee.
    result = @accelerate aggressive=true _offset_ordinary_call(a)
    @test Array(result) == Array(a) .- 3
    @test length(cuNumeric._BROADCAST_VARIANT_CACHE) == 3
    @test count(p -> p isa cuNumeric.AdaptiveBroadcastPlan,
        values(cuNumeric._BROADCAST_PLAN_CACHE)) == 1
end

@testset "Fixed and adaptive GPU offsets, ranks 1–6" begin
    for N in 1:6
        dims = ntuple(d -> d == 2 ? 1 : 3, N)
        host = reshape(Float32.(1:prod(dims)), dims)
        a = NDArray(host)
        for policy in (
            cuNumeric.FixedOffsets{Int64}(),
            cuNumeric.FixedOffsets{Int32}(),
            cuNumeric.AdaptiveOffsets(),
        )
            result = cuNumeric._materialize_with_policy(
                policy, Base.broadcasted(_offset_test_op, a, 2.0f0)
            )
            @test Array(result) == host .* 2 .+ 1
            # Alias protection: the destination is also a repeated operand.
            dest = copy(a)
            tree = Base.broadcasted(_offset_test_pair, dest, dest)
            cuNumeric._materialize_with_policy!(policy, dest, tree)
            @test Array(dest) == host .* 3
        end
        # Both multi-output factories and the macro's adaptive policy work at every rank.
        outputs = @accelerate aggressive=true begin
            x = a .* 2.0f0
            y = x .+ 1.0f0
            (x, y)
        end
        @test Array(outputs[1]) == host .* 2
        @test Array(outputs[2]) == host .* 2 .+ 1
        d1, d2 = similar(a), similar(a)
        @accelerate aggressive=true begin
            d1 .= a .* 2.0f0 .+ 1.0f0
            d2 .= a .* 3.0f0 .+ 2.0f0
        end
        @test Array(d1) == host .* 2 .+ 1
        @test Array(d2) == host .* 3 .+ 2
        # Exercise the shared sibling launch for every rank, independent of
        # whether the source-level update grouping chooses to merge this block.
        cuNumeric.copyto_fused_siblings!(
            cuNumeric.AdaptiveOffsets(),
            (d1, d2),
            (
                Base.broadcasted(_offset_test_op, a, 4.0f0),
                Base.broadcasted(_offset_test_op, a, 5.0f0),
            ),
        )
        @test Array(d1) == host .* 4 .+ 1
        @test Array(d2) == host .* 5 .+ 1
    end
end

@testset "Adaptive slices, fallback and structs" begin
    host = reshape(Float32.(1:120), 10, 12)
    a = NDArray(host)
    for r in (1:5, 3:7)
        slice = a[r, 2:7]
        result = @accelerate aggressive=true _offset_test_op.(slice, 2.0f0)
        @test Array(result) == host[r, 2:7] .* 2 .+ 1
    end
    v = NDArray(Int32.(ones(10, 12)))
    @test Array(@accelerate aggressive=true _offset_test_pair.(a, v)) == host .* 2 .+ 1
    # Mismatched shapes use the preexisting unfused broadcast path.
    row = NDArray(reshape(Float32.(1:12), 1, 12))
    @test Array(@accelerate aggressive=true a .+ row) == host .+ reshape(Float32.(1:12), 1, 12)
    chain = @accelerate aggressive=true begin
        first = a .* 2.0f0
        second = first .+ row
        (first, second)
    end
    @test Array(chain[1]) == host .* 2
    @test Array(chain[2]) == host .* 2 .+ reshape(Float32.(1:12), 1, 12)
    empty = NDArray(zeros(Float32, 0, 3))
    @test size(@accelerate aggressive=true empty .* 2.0f0 .+ 1.0f0) == (0, 3)
    scalar = NDArray(fill(2.0f0))
    @test Array(@accelerate aggressive=true scalar .* 2.0f0 .+ 1.0f0) == fill(5.0f0)
    # Output/input overlap prevents sibling fusion; each fallback keeps its policy.
    x = NDArray(Float32.(1:8))
    y = similar(x)
    @accelerate aggressive=true begin
        x .= x .* 2.0f0 .+ 1.0f0
        y .= x .* 3.0f0 .+ 2.0f0
    end
    @test Array(y) == (Float32.(1:8) .* 2 .+ 1) .* 3 .+ 2
    records = NDArray(OffsetTestRecord.(Float32.(1:8)))
    before = length(cuNumeric._BROADCAST_VARIANT_CACHE)
    result = @accelerate aggressive=true _offset_record_op.(records)
    @test length(cuNumeric._BROADCAST_VARIANT_CACHE) == before + 1
    # Host transfer projects record fields with its own wide broadcast kernel.
    @test getfield.(Array(result), :x) == 2 .* Float32.(1:8)
end
