using StructArrays

# Struct element storage needs the fused GPU broadcast kernel. Without a GPU or
# with fusion disabled, every operation that touches element data must raise an
# ArgumentError instead of reaching the unfused path or a missing GPU variant.

struct GuardPair
    a::Float32
    b::Int32
end

_guard_pair(x) = GuardPair(Float32(x + 1), Int32(2x))
_guard_pair_b(p::GuardPair) = p.b

if cuNumeric._struct_kernel_available()
    @testset "Struct guards (skipped: struct kernel available)" begin
        @test_skip false
    end
else
    @testset "Struct storage without the fused kernel" begin
        template = cuNumeric.zeros(Int64, 6)
        value = GuardPair(3.0f0, Int32(4))
        arr = similar(template, GuardPair, (6,))
        other = similar(template, GuardPair, (6,))
        input = NDArray(collect(0:5))
        empty_arr = similar(template, GuardPair, (0,))
        try
            # Allocation, fills, scalar writes and views need no kernel.
            @test arr isa NDArray{GuardPair,1}
            @test fill!(arr, value) === arr
            cuNumeric.allowscalar() do
                arr[2] = GuardPair(9.0f0, Int32(9))
            end
            view = arr[2:4]
            @test size(view) == (3,)
            cuNumeric.destroy!(view)

            # Empty arrays have no elements to move.
            @test isempty(Array(empty_arr))
            @test isempty(Array(NDArray(GuardPair[])))

            @test_throws ArgumentError NDArray([value, value])
            @test_throws ArgumentError Array(arr)
            @test_throws ArgumentError cuNumeric.allowscalar(() -> arr[1])
            @test_throws ArgumentError copy(arr)
            @test_throws ArgumentError copyto!(other, arr)
            @test_throws ArgumentError cuNumeric.reshape(arr, 2, 3)
            @test_throws ArgumentError (arr == other)
            @test_throws ArgumentError (arr .= _guard_pair.(input))
            @test_throws ArgumentError _guard_pair_b.(arr)

            err = try
                Array(arr)
            catch e
                e
            end
            @test occursin("requires GPU broadcast fusion", sprint(showerror, err))
        finally
            foreach(cuNumeric.destroy!, (template, arr, other, input, empty_arr))
        end
    end

    @testset "StructArray broadcast without the fused kernel" begin
        previous_experimental = get(task_local_storage(), :Experimental, false)
        input = NDArray(collect(0:2))
        dest = StructArray{GuardPair}((a=cuNumeric.zeros(Float32, 3), b=cuNumeric.zeros(Int32, 3)))
        try
            cuNumeric.Experimental(true)
            @test_throws ArgumentError (dest .= _guard_pair.(input))
            @test all(iszero, Array(dest.a))
            # Field storage is ordinary NDArrays.
            @test Array(dest.a .+ 1.0f0) == ones(Float32, 3)
        finally
            cuNumeric.Experimental(previous_experimental)
            foreach(cuNumeric.destroy!, (input, dest.a, dest.b))
        end
    end
end
