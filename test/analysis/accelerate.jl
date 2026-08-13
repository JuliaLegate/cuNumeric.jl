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

    # The multi-output launch used by the `begin` chain form is GPU-only.
    if cuNumeric.HAS_CUDA
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
        end
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
        end
    end
end
