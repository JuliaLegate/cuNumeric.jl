using Test

const IBF = cuNumeric.InterBroadcastFusion
const SU = cuNumeric.ScopingUtils

@testset "Nonadjacent producer elimination" begin
    ex = IBF.rewrite_scope(quote
        t = a .+ 1f0
        r = b .* 2f0
        out .= t .+ r
    end)
    @test !(:t in SU.walk_symbols(ex))
    @test !(:r in SU.walk_symbols(ex))

    # Expanding t into s must retain a as a dependency of s.
    ex = IBF.rewrite_scope(quote
        t = a .+ 1f0
        s = t .* 2f0
        a = b .+ 2f0
        s .+ a
    end)
    @test !(:t in SU.walk_symbols(ex))
    @test :s in SU.walk_symbols(ex)

    # Calls can mutate inputs without a bang suffix, including inside a broadcast.
    for barrier in (:(touch(a)), :(unused = touch.(a)), :(a[1] = 9f0))
        ex = IBF.rewrite_scope(quote
            t = a .+ 1f0
            $barrier
            out .= t .* 2f0
        end)
        @test :t in SU.walk_symbols(ex)
    end
    ex = IBF.rewrite_scope(quote
        t = a .+ 1f0
        r = b .* 2f0
        t .+ r
    end; protected=Set([:t]))
    @test :t in SU.walk_symbols(ex)

    ex = IBF.rewrite_scope(quote
        t = a .+ 1f0
        out[touch(a)] .= t .* 2f0
    end)
    @test :t in SU.walk_symbols(ex)
    ex = IBF.rewrite_scope(quote
        t = sin.(a)
        sin = cos
        t .+ 1f0
    end)
    @test :t in SU.walk_symbols(ex)

    # A compact Gray-Scott-shaped sequence: four producers and two array writes.
    body = quote
        F_u = u .* v
        F_v = u .+ v
        u_lap = u .* 2f0
        v_lap = v .* 3f0
        u_new[:] = F_u .+ u_lap
        v_new[:] = F_v .+ v_lap
        nothing
    end
    checks = Tuple{Symbol,Symbol}[]
    ex = IBF.rewrite_scope(body; guard_roots=Set([:u, :v, :u_new, :v_new]),
        alias_checks=checks)
    @test all(s -> !(s in SU.walk_symbols(ex)), (:F_u, :F_v, :u_lap, :v_lap))
    @test Set(checks) == Set([(:u_new, :u), (:u_new, :v)])
    fallback = IBF.rewrite_scope(body)
    @test :F_v in SU.walk_symbols(fallback)
    @test :v_lap in SU.walk_symbols(fallback)

    # Guards cannot be hoisted past unknown calls that could change storage.
    prefixed = Expr(:block, :(prepare(u_new, u)), SU._scope_statements(body)...)
    checks = Tuple{Symbol,Symbol}[]
    IBF.rewrite_scope(prefixed; guard_roots=Set([:u, :v, :u_new, :v_new]), alias_checks=checks)
    @test isempty(checks)
end

@accelerate function _acc_two_updates!(u, v, u_new, v_new)
    F_u = u .* v
    F_v = u .+ v
    u_lap = u .* 2f0
    v_lap = v .* 3f0
    u_new[:] = F_u .+ u_lap
    v_new[:] = F_v .+ v_lap
    nothing
end
function _plain_two_updates!(u, v, u_new, v_new)
    F_u = u .* v
    F_v = u .+ v
    u_lap = u .* 2f0
    v_lap = v .* 3f0
    u_new[:] = F_u .+ u_lap
    v_new[:] = F_v .+ v_lap
    nothing
end

@testset "Guarded fusion preserves shared inputs" begin
    for make in (identity, NDArray), alias in (:none, :u, :v, :shifted)
        host = Float32[1, 2, 3, 4, 5]
        hu = view(host, 1:4)
        hv = Float32[5, 6, 7, 8]
        ho = alias === :none ? zeros(Float32, 4) :
             alias === :u ? hu : alias === :v ? hv : view(host, 2:5)
        hout = zeros(Float32, 4)
        parent = make(copy(host))
        u = view(parent, 1:4)
        v = make(copy(hv))
        unew = alias === :none ? make(zeros(Float32, 4)) :
               alias === :u ? u : alias === :v ? v : view(parent, 2:5)
        vnew = make(zeros(Float32, 4))
        _plain_two_updates!(hu, hv, ho, hout)
        @test _acc_two_updates!(u, v, unew, vnew) === nothing
        @test Array(unew) == ho
        @test Array(vnew) == hout
        @test Array(parent) == host
        if make === NDArray
            foreach(cuNumeric.destroy!, (unew, vnew, u, v, parent))
        end
    end
end

@testset "Transitive rebindings and unknown calls remain barriers" begin
    @accelerate function transitive(a, b)
        t = a .+ 1f0
        s = t .* 2f0
        a = b .+ 2f0
        s .+ a
    end
    function touch(a)
        fill!(a, 9f0)
        nothing
    end
    @accelerate function unknown_effect(a)
        t = a .+ 1f0
        touch(a)
        t .* 2f0
    end
    for make in (identity, NDArray)
        @test Array(transitive(make(Float32[1, 2]), make(Float32[10, 20]))) == Float32[16, 28]
        @test Array(unknown_effect(make(Float32[1, 2]))) == Float32[4, 6]
    end
end
