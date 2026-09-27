using Test

@testset "Native n-ary broadcasts reuse the destination" begin
    a = NDArray(Float32[1, 2, 3, 4])
    b = NDArray(Float32[2, 3, 4, 5])
    c = NDArray(Float32[3, 4, 5, 6])
    dest = cuNumeric.zeros(Float32, 4)
    for f in (+, *)
        for args in ((a, b, c), (a, b, c, a), (2f0, 3f0, a),
                     (Base.broadcasted(+, a, 1f0), b, c))
            bc = Base.Broadcast.instantiate(Base.broadcasted(f, args...))
            host_args = map(args) do arg
                arg isa NDArray && return Array(arg)
                arg isa Base.Broadcast.Broadcasted && return Array(a) .+ 1f0
                return arg
            end
            expected = f.(host_args...)
            # Exercise the native route regardless of the fusion preference.
            result = cuNumeric.unravel_broadcast_tree(bc, dest)
            @test result === dest
            @test Array(dest) == expected
            @test Array(a) == Float32[1, 2, 3, 4]
            @test Array(b) == Float32[2, 3, 4, 5]
            @test Array(c) == Float32[3, 4, 5, 6]

            allocated = cuNumeric.unravel_broadcast_tree(bc)
            @test allocated !== dest
            @test Array(allocated) == expected
            cuNumeric.destroy!(allocated)
        end
        # The destination can be read both before and during the final operation.
        fill!(dest, 2f0)
        bc = Base.Broadcast.instantiate(Base.broadcasted(f, dest, a, dest))
        @test cuNumeric._unfused_into!(dest, bc) === dest
        @test Array(dest) == f.(2f0, Float32[1, 2, 3, 4], 2f0)
    end
    foreach(cuNumeric.destroy!, (a, b, c, dest))
end

@testset "Native n-ary broadcasts preserve overlap and conversion guards" begin
    for f in (+, *)
        parent = NDArray(Float32[1, 2, 3, 4, 5])
        source = view(parent, 1:4)
        dest = view(parent, 2:5)
        a = cuNumeric.fill(2f0, 4)
        b = cuNumeric.fill(3f0, 4)
        bc = Base.Broadcast.instantiate(Base.broadcasted(f, a, b, source))
        result = cuNumeric.unravel_broadcast_tree(bc, dest)
        @test result !== dest
        @test Array(parent) == Float32[1, 2, 3, 4, 5]
        @test cuNumeric._copyto_unfused!(dest, result) === dest
        @test Array(parent) == vcat(1f0, f.(2f0, 3f0, Float32[1, 2, 3, 4]))
        foreach(cuNumeric.destroy!, (source, dest, parent, a, b))
    end

    ints = NDArray(Int32[1, 2, 3, 4])
    dest = cuNumeric.zeros(Float32, 4)
    alias = view(dest, 1:2)
    bc = Base.Broadcast.instantiate(Base.broadcasted(+, ints, Int32(2), Int32(3)))
    result = cuNumeric.unravel_broadcast_tree(bc, dest)
    @test result !== dest
    @test eltype(result) === Int32
    @test cuNumeric._copyto_unfused!(dest, result) === dest
    @test Array(dest) == Float32[6, 7, 8, 9]
    @test Array(alias) == Float32[6, 7]
    foreach(cuNumeric.destroy!, (alias, dest, ints))
end
