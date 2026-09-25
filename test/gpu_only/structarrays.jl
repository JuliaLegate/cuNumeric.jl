using StructArrays

struct BroadcastPair
    a::Float64
    b::Float64
end

struct MixedFields
    a::Float32
    b::Int32
    c::Bool
end

broadcast_pair(x) = BroadcastPair(x + 1.0, x + 2.0)
swap_pair(a, b) = BroadcastPair(b + 10.0, a + 20.0)
mixed_fields(x) = MixedFields(x + 1.0f0, Int32(2x), x >= 2.0f0)

@testset "StructArray broadcast (experimental)" begin
    previous_experimental = get(task_local_storage(), :Experimental, false)
    input = NDArray(reshape(Int64[0, 1, 2], 3, 1))
    dest = StructArray{BroadcastPair}((
        a=cuNumeric.zeros(Float64, 3, 1),
        b=cuNumeric.zeros(Float64, 3, 1),
    ))
    mixed_input = NDArray(reshape(Float32[0, 1, 2, 3], 2, 2))
    mixed_dest = StructArray{MixedFields}((
        a=cuNumeric.zeros(Float32, 2, 2),
        b=cuNumeric.zeros(Int32, 2, 2),
        c=cuNumeric.zeros(Bool, 2, 2),
    ))
    wrong_shape = NDArray(reshape(Int64[0, 1, 2], 1, 3))
    empty_input = cuNumeric.zeros(Int64, 0)
    empty_dest = StructArray{BroadcastPair}((
        a=cuNumeric.zeros(Float64, 0),
        b=cuNumeric.zeros(Float64, 0),
    ))
    host_dest = StructArray{BroadcastPair}((a=zeros(3, 1), b=zeros(3, 1)))

    try
        @test Base.get_extension(cuNumeric, :cuNumericStructArraysExt) !== nothing
        cuNumeric.Experimental(false)
        disabled_error = try
            dest .= broadcast_pair.(input)
            nothing
        catch err
            err
        end
        @test disabled_error isa ArgumentError
        @test occursin("Experimental features are disabled", sprint(showerror, disabled_error))
        @test_throws ArgumentError (empty_dest .= broadcast_pair.(empty_input))
        @test all(iszero, Array(dest.a))
        @test all(iszero, Array(dest.b))

        cuNumeric.Experimental(true)
        @test (dest .= broadcast_pair.(input)) === dest
        @test vec(Array(dest.a)) == [1.0, 2.0, 3.0]
        @test vec(Array(dest.b)) == [2.0, 3.0, 4.0]

        # Both outputs must read the old values of both destination fields.
        dest .= swap_pair.(dest.a, dest.b)
        @test vec(Array(dest.a)) == [12.0, 13.0, 14.0]
        @test vec(Array(dest.b)) == [21.0, 22.0, 23.0]

        # Different field types and a nested broadcast share one struct result.
        mixed_dest .= mixed_fields.(mixed_input .+ 1.0f0)
        @test Array(mixed_dest.a) == reshape(Float32[2, 3, 4, 5], 2, 2)
        @test Array(mixed_dest.b) == reshape(Int32[2, 4, 6, 8], 2, 2)
        @test Array(mixed_dest.c) == reshape(Bool[false, true, true, true], 2, 2)

        @test_throws DimensionMismatch (dest .= broadcast_pair.(wrong_shape))
        @test_throws ArgumentError (host_dest .= broadcast_pair.(input))
        @test (empty_dest .= broadcast_pair.(empty_input)) === empty_dest
        cuNumeric.Experimental(false)
        @test_throws ArgumentError (dest .= broadcast_pair.(input))
    finally
        cuNumeric.Experimental(previous_experimental)
        for arr in (dest, mixed_dest, empty_dest)
            foreach(cuNumeric.destroy!, Tuple(StructArrays.components(arr)))
        end
        foreach(cuNumeric.destroy!, (input, mixed_input, wrong_shape, empty_input))
    end
end
