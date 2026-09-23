using StructArrays

struct BroadcastPair
    a::Float64
    b::Float64
end

broadcast_pair(x) = BroadcastPair(x + 1.0, x + 2.0)

@testset "StructArray broadcast" begin
    input = NDArray(reshape(Int64[0, 1, 2], 3, 1))
    dest = StructArray{BroadcastPair}((
        a=cuNumeric.zeros(Float64, 3, 1),
        b=cuNumeric.zeros(Float64, 3, 1),
    ))
    try
        dest .= broadcast_pair.(input)
        @test vec(Array(dest.a)) == [1.0, 2.0, 3.0]
        @test vec(Array(dest.b)) == [2.0, 3.0, 4.0]

        # The first field is both input and output; all fields must read its old value.
        dest .= broadcast_pair.(dest.a)
        @test vec(Array(dest.a)) == [2.0, 3.0, 4.0]
        @test vec(Array(dest.b)) == [3.0, 4.0, 5.0]

    finally
        foreach(cuNumeric.destroy!, Tuple(StructArrays.components(dest)))
        cuNumeric.destroy!(input)
    end
end
