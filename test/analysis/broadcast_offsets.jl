@testset "Fixed element-offset arithmetic" begin
    for N in 1:6
        dims = ntuple(d -> d == 2 ? 1 : 3, N)
        for strides in (ntuple(d -> 3^(d - 1), N), ntuple(d -> isodd(d) ? -d : 0, N))
            for I in CartesianIndices(dims)
                reference = sum((I[d] - 1) * strides[d] for d in 1:N)
                linear = LinearIndices(dims)[I]
                for T in (Int32, Int64)
                    policy = cuNumeric.FixedOffsets{T}()
                    @test cuNumeric._strided_elem_offset(policy, strides, I) == reference
                    @test cuNumeric._strided_elem_offset(policy, dims, strides, linear) == reference
                end
            end
        end
    end
    limit = Int(typemax(Int32))
    for offset in (limit - 1, limit)
        @test cuNumeric._strided_elem_offset(
            cuNumeric.FixedOffsets{Int32}(), (offset,), CartesianIndex(2)
        ) == offset
    end
    @test cuNumeric._strided_elem_offset(
        cuNumeric.FixedOffsets{Int64}(), (limit + 1,), CartesianIndex(2)
    ) == limit + 1
    # Decoding remains wide even when the volume exceeds INT32_MAX. The
    # zero stride makes the final element offset safe to narrow.
    dims = (2, limit)
    @test cuNumeric._strided_elem_offset(cuNumeric.FixedOffsets{Int32}(), dims, (1, 0), 2limit) == 1
end

@testset "Adaptive plan ABI" begin
    task = cuNumeric.CUDATask("wide", (Int64, Float32))
    wide = cuNumeric.CompiledBroadcast(task, (Int64(11),), 128)
    narrow = cuNumeric.CompiledBroadcast(
        cuNumeric.CUDATask("narrow", task.argtypes), (Int64(21),), 256
    )
    plan = cuNumeric.AdaptiveBroadcastPlan(wide, narrow)
    @test plan.wide.ctx == (11,)
    @test plan.narrow.ctx == (21,)
    @test plan.wide.threads == 128
    @test plan.narrow.threads == 256
    @test_throws ArgumentError cuNumeric.AdaptiveBroadcastPlan(
        cuNumeric.CompiledBroadcast(task, UInt8[1], 128),
        cuNumeric.CompiledBroadcast(task, UInt8[1, 2], 256),
    )
    @test_throws ArgumentError cuNumeric.AdaptiveBroadcastPlan(wide,
        cuNumeric.CompiledBroadcast(cuNumeric.CUDATask("bad", (Int32,)), narrow.ctx, 256))
    @test_throws ArgumentError cuNumeric.AdaptiveBroadcastPlan(wide,
        cuNumeric.CompiledBroadcast(narrow.cuda_task, (Int32(21), Int32(22)), 256))
end

@testset "Lexical broadcast policy" begin
    forms = (
        :(function policy_function(a)
            ;return a .* 2 .+ 1;
        end),
        :(
            begin
                x = a .* 2
                y = x .+ 1
                (x, y)
            end
        ),
        :(
            let x = a .* 2
                x .+ 1
            end
        ),
        :(a .* 2 .+ 1),
    )
    for ex in forms, aggressive in (false, true)
        lowered = string(cuNumeric._accelerate_expand(ex, @__MODULE__; aggressive))
        @test occursin(aggressive ? "AdaptiveOffsets" : "FixedOffsets", lowered)
        @test !occursin(aggressive ? "FixedOffsets" : "AdaptiveOffsets", lowered)
    end
    a = Float32[1, 2, 3]
    @test (@accelerate aggressive=true a .* 2 .+ 1) == a .* 2 .+ 1
    @test (@accelerate aggressive=true let x = a .* 2
        x .+ 1
    end) == a .* 2 .+ 1
    chain = @accelerate aggressive=true begin
        x = a .* 2
        y = x .+ 1
        (x, y)
    end
    @test chain == (a .* 2, a .* 2 .+ 1)
    @accelerate aggressive=true function policy_update!(dest, a)
        dest .+= a .* 2
        dest .*= 3
        return dest
    end
    for make in (identity, NDArray)
        @test Array(policy_update!(make(copy(a)), make(a))) == a .* 9
    end
    @accelerate aggressive=true function policy_return(a)
        return a .* 2 .+ 1
    end
    for make in (identity, NDArray)
        @test Array(policy_return(make(a))) == a .* 2 .+ 1
    end
    # A destination call is evaluated once; an ordinary callee is unchanged.
    calls = Ref(0)
    dest = zeros(Float32, 3)
    destination() = (calls[] += 1; dest)
    @accelerate aggressive=true destination() .+= a .* 2
    @test calls[] == 1
    @test dest == 2a
    ex = cuNumeric._accelerate_expand(:(unrelated(a)), @__MODULE__; aggressive=true)
    @test !occursin("AdaptiveOffsets", string(ex))
    @test (@accelerate aggressive=true a .^ 2) == a .^ 2
    @test (@accelerate aggressive=true sin.(a)) ≈ sin.(a)
end
