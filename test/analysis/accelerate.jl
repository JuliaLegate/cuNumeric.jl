#= Copyright 2026 Northwestern University,
 *                   Carnegie Mellon University University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author(s): David Krasowska <krasow@u.northwestern.edu>
 *            Ethan Meitz <emeitz@andrew.cmu.edu>
=#

# Coverage of the four `@accelerate` forms and their scope contracts:
#   @accelerate function f(...) ... end   -> function scope, frees non-returned
#   @accelerate begin ... end             -> 1:1 Julia scope, bindings stay alive
#   @accelerate let ... end               -> hard scope, combine + free non-returned
#   @accelerate expr                      -> materialized result, temps freed

using InteractiveUtils: code_typed

@testset "@accelerate respects rebindings and alias writes" begin
    @accelerate function _acc_rebind(a, b)
        t = a .+ 1.0f0
        a = b .+ 2.0f0
        t .+ a
    end
    @accelerate function _acc_aliaswrite(a, b)
        t = a .+ 1.0f0
        b[1] = 9.0f0
        t .+ 0.0f0
    end
    for make in (identity, NDArray)
        @test Array(_acc_rebind(make(Float32[1]), make(Float32[10]))) == Float32[14]
        a = make(Float32[1, 2])
        @allowscalar result = _acc_aliaswrite(a, a)
        @test Array(result) == Float32[2, 3]
    end
    # Adjacent chains still inline their single-use intermediates.
    ex = cuNumeric.InterBroadcastFusion.rewrite_scope(quote
        t = a .+ 1.0f0
        u = t .* 2.0f0
        u .+ 3.0f0
    end)
    @test !(:t in cuNumeric.ScopingUtils.walk_symbols(ex))
    @test !(:u in cuNumeric.ScopingUtils.walk_symbols(ex))
end

@testset "@accelerate aggressive=true merges sibling updates" begin
    step = quote
        un[2:(end - 1)] .= u[2:(end - 1)] .* 2.0f0 .+ v[2:(end - 1)]
        vn[2:(end - 1)] .= v[2:(end - 1)] .* 3.0f0 .- u[1:(end - 2)]
    end
    fn = Expr(:function, :(_acc_siblings(u, v, un, vn)), step)
    merged(ex) = occursin("copyto_fused_siblings!", string(ex))
    @test !merged(cuNumeric._accelerate_expand(fn, @__MODULE__))
    # Only the fusion pipeline merges; without fusion `aggressive` is a no-op.
    @test merged(cuNumeric._accelerate_expand(fn, @__MODULE__; aggressive=true)) ==
        cuNumeric.FUSE_BROADCAST_EXPRS
    @test_throws ErrorException cuNumeric._accelerate_options((:(fast = true),))
    @test cuNumeric._accelerate_options(()).aggressive               # on by default
    @test !cuNumeric._accelerate_options((:(aggressive = false),)).aggressive
    # Hoisted constants such as `Int8(2)` do not block the merge.
    consts = Expr(:function, :(_acc_consts(P)), quote
        P[3:4, :] .= P[1:2, :] .* Int8(2)
        P[5:6, :] .= P[1:2, :] .- Int8(1)
    end)
    @test merged(cuNumeric._accelerate_expand(consts, @__MODULE__; aggressive=true)) ==
        cuNumeric.FUSE_BROADCAST_EXPRS

    @accelerate aggressive=true function _acc_siblings(u, v, un, vn)
        un[2:(end - 1)] .= u[2:(end - 1)] .* 2.0f0 .+ v[2:(end - 1)]
        vn[2:(end - 1)] .= v[2:(end - 1)] .* 3.0f0 .- u[1:(end - 2)]
    end
    function reference!(u, v, un, vn)
        un[2:(end - 1)] .= u[2:(end - 1)] .* 2.0f0 .+ v[2:(end - 1)]
        vn[2:(end - 1)] .= v[2:(end - 1)] .* 3.0f0 .- u[1:(end - 2)]
    end
    N = 64
    u0, v0 = rand(Float32, N), rand(Float32, N)
    for make in (identity, NDArray)
        un, vn = make(zeros(Float32, N)), make(zeros(Float32, N))
        _acc_siblings(make(u0), make(v0), un, vn)
        ref_un, ref_vn = zeros(Float32, N), zeros(Float32, N)
        reference!(u0, v0, ref_un, ref_vn)
        @test Array(un) ≈ ref_un
        @test Array(vn) ≈ ref_vn

        # Aliased outputs fall back.
        a, b = make(copy(u0)), make(copy(v0))
        _acc_siblings(a, b, a, b)
        ra, rb = copy(u0), copy(v0)
        reference!(ra, rb, ra, rb)
        @test Array(a) ≈ ra
        @test Array(b) ≈ rb
    end
    if cuNumeric.FUSE_BROADCAST_EXPRS && cuNumeric._has_gpu_target()
        u, v = NDArray(u0), NDArray(v0)
        dests = (NDArray(zeros(Float32, N))[2:(end - 1)], NDArray(zeros(Float32, N))[2:(end - 1)])
        bcs = (Base.broadcasted(+, Base.broadcasted(*, u[2:(end - 1)], 2.0f0), v[2:(end - 1)]),
            Base.broadcasted(-, Base.broadcasted(*, v[2:(end - 1)], 3.0f0), u[1:(end - 2)]))
        @test !isnothing(cuNumeric._sibling_segments(dests, bcs))      # one launch
        mismatched = (dests[1], NDArray(zeros(Float32, N - 4)))      # shapes differ
        @test isnothing(cuNumeric._sibling_segments(mismatched, bcs))
    end
end

@testset "@accelerate let guards read-only free variables like arguments" begin
    # `t` may move past the write to `y` only when `y` and `x` do not overlap.
    function run_let(x, y)
        return @accelerate let
            t = x .+ 1.0f0
            y[1:2] .= x[3:4] .* 2.0f0
            t .* 3.0f0
        end
    end
    function reference(x, y)
        t = x .+ 1.0f0
        y[1:2] .= x[3:4] .* 2.0f0
        return t .* 3.0f0
    end
    for make in (identity, NDArray)
        x0 = Float32[1, 2, 3, 4]
        @test Array(run_let(make(copy(x0)), make(zeros(Float32, 4)))) ==
            reference(copy(x0), zeros(Float32, 4))
        x = make(copy(x0))                                   # y aliases x
        @test Array(run_let(x, x)) == (xr=copy(x0); reference(xr, xr))
    end
    body = quote
        un[2:(end - 1)] .= u[2:(end - 1)] .* 2.0f0 .+ v[2:(end - 1)]
        vn[2:(end - 1)] .= v[2:(end - 1)] .* 3.0f0 .- u[1:(end - 2)]
    end
    expanded = cuNumeric._accelerate_block_hard(Expr(:let, body), @__MODULE__; aggressive=true)
    @test occursin("copyto_fused_siblings!", string(expanded)) == cuNumeric.FUSE_BROADCAST_EXPRS
end

@testset "@accelerate — four forms" begin
    T = Float32
    N = 64
    _nd(v) = @allowscalar NDArray(v)
    approx(x, ref) = isapprox(Array(x), ref; rtol=1.0f-4)
    ja = my_rand(T, N)
    jb = my_rand(T, N)

    @testset "1. function form" begin
        @accelerate function _acc_fsq(a, b)
            c = a .* b
            return c .^ 2
        end
        a = _nd(ja)
        b = _nd(jb)
        @test approx(_acc_fsq(a, b), (ja .* jb) .^ 2)
        # Arguments are caller-owned: a second call on the same inputs still works.
        @test approx(_acc_fsq(a, b), (ja .* jb) .^ 2)
    end

    @testset "3. let form (hard scope)" begin
        function _acc_let(a, b)
            s = @accelerate let
                r = a .+ b
                s = r .* T(2)
                s
            end
            return s, @isdefined(r)
        end
        s, r_leaked = _acc_let(_nd(ja), _nd(jb))
        @test approx(s, (ja .+ jb) .* T(2))
        @test r_leaked == false          # `r` must not escape the let scope
    end

    @testset "4. expr form" begin
        a = _nd(ja)
        b = _nd(jb)
        res = @accelerate (a .+ b) .^ 2
        @test res isa NDArray
        @test approx(res, (ja .+ jb) .^ 2)
    end

    @testset "2. begin form (bindings stay alive)" begin
        function _acc_begin(a, b)
            q = @accelerate begin
                p = a .* b
                q = p .+ one(T)
                q
            end
            return p, q              # both must be defined in this scope
        end
        p, q = _acc_begin(_nd(ja), _nd(jb))
        @test approx(p, ja .* jb)
        @test approx(q, (ja .* jb) .+ one(T))

        # A nested `let` keeps its intermediate private while the outer block
        # can consume and return the value it produces.
        a = _nd(ja)
        b = _nd(jb)
        one_t = one(T)
        shifted, x = @accelerate begin
            shifted = let
                product = @. a * b
                @. product + one_t
            end
            x = @. shifted * 2
            (shifted, x)
        end
        @test approx(shifted, (ja .* jb) .+ one(T))
        @test approx(x, ((ja .* jb) .+ one(T)) .* 2)
    end

    @testset "expansion contracts (white-box)" begin
        expand(ex) = cuNumeric._accelerate_expand(ex, @__MODULE__)
        hasfree(ex) = occursin("maybe_insert_delete", string(expand(ex)))

        # Scope shape per form.
        @test expand(:(function f(a)
            ;c = a .* a;
            c .^ 2;
        end)).head === :function
        @test expand(:(
            begin
                C .= a[2:end] .+ b[2:end]
            end
        )).head === :block
        @test expand(:(
            let
                r = a .+ b;
                r .* 2
            end
        )).head === :let

        # Slices are freed in every non-`let` form (uniform cleanup).
        @test hasfree(:(function f(a)
            ;s = a[2:end];
            s .+ 1;
        end))
        @test hasfree(:(
            begin
                C .= a[2:end] .+ b[2:end]
            end
        ))

        # Compound dotted assignments must remain one broadcast tree. Hoisting
        # their RHS would add a full-size temporary and a second GPU launch.
        compound = string(expand(:(function update!(x, alpha, p)
            x .+= alpha .* p
            x
        end)))
        @test occursin("x .+= alpha .* p", compound)
        @test !occursin(r"tmp\d+ = alpha \.\* p", compound)

        if cuNumeric.FUSE_BROADCAST_EXPRS
            # A same-shape chain fuses into one multi-output launch and still
            # frees the hoisted slice temporaries.
            mo = string(expand(:(
                begin
                    p = a[2:end] .* b[2:end]
                    q = p .+ 1
                    q
                end
            )))
            @test occursin("copyto_fused_multi_alloc!", mo)
            @test occursin("maybe_insert_delete", mo)
        else
            # Multi-output fusion is GPU-only; CPU expansion must use the
            # ordinary broadcast path even when fusion is enabled in preferences.
            cpu = string(expand(:(
                begin
                    p = a .* b
                    q = p .+ 1
                    q
                end
            )))
            @test !occursin("copyto_fused_multi_alloc!", cpu)
        end
    end

    @testset "multi-output segment runner is fully unrolled" begin
        # GPU compilation requires every chained segment call to be statically
        # dispatched. This three-segment shape crossed Julia 1.10's recursive
        # inference limit when `_run_segments` recursed over `Base.tail`.
        segs = (
            (+, (cuNumeric.RuntimeBroadcastArg{1}(), cuNumeric.RuntimeBroadcastArg{2}())),
            (*, (cuNumeric.LocalBroadcastArg{1}(), cuNumeric.RuntimeBroadcastArg{1}())),
            (^, (cuNumeric.LocalBroadcastArg{2}(), cuNumeric.RuntimeBroadcastArg{3}())),
        )
        outs = ntuple(_ -> zeros(T, 2, 2), 3)
        runtime_args = (ones(T, 2, 2), ones(T, 2, 2), 2)

        @test @inferred(
            cuNumeric._run_segments(
                segs, outs, runtime_args, (), (), CartesianIndex(1, 1)
            )
        ) === nothing
        @test getindex.(outs, Ref(CartesianIndex(1, 1))) == (T(2), T(2), T(4))

        argtypes = (
            typeof(segs),
            typeof(outs),
            typeof(runtime_args),
            Tuple{},
            Tuple{},
            CartesianIndex{2},
        )
        typed = only(code_typed(cuNumeric._run_segments, argtypes; optimize=true)).first
        @test !occursin(
            "_run_segments", sprint(show, MIME("text/plain"), typed)
        )
    end
end
