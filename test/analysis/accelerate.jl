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
        shifted, x = @accelerate begin
            shifted = let
                product = @. a * b
                @. product + one(T)
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

        if cuNumeric.FUSE_BROADCAST_EXPRS && cuNumeric.HAS_CUDA
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
