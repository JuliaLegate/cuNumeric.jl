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

function run_test(op, op_scope, FT, N)
    a = cuNumeric.rand(FT, (N, N))
    b = cuNumeric.rand(FT, (N, N))
    c_scoped = cuNumeric.zeros(FT, (N, N))

    c_base = op(a, b)

    cuNumeric.disable_gc!(; verbose=false)
    op_scope(a, b, c_scoped)
    cuNumeric.init_gc!()

    return c_base, c_scoped
end

function run_slice_test(op, op_scoped, FT, N; f=0.04, k=0.06, dx=1.0)
    u = cuNumeric.rand(FT, (N, N))
    v = cuNumeric.rand(FT, (N, N))

    scoped = cuNumeric.zeros(FT, (N-2, N-2))
    args = (f=FT(f), k=FT(k), dx=FT(dx))

    base = op(u, v, args)

    cuNumeric.disable_gc!(; verbose=false)
    op_scoped(u, v, scoped, args)
    cuNumeric.init_gc!()

    return base, scoped
end

binary_scope(op) = (a, b, out) -> @analyze_lifetimes out[:, :] = op(a, b)
slice_scope(op) = (u, v, out, args) -> @analyze_lifetimes out[:, :] = op(u, v, args)

const OPS = Dict(
    :add => (+),
    :negate_add => ((a, b) -> -a + b),
    :sub => (-),
    :mul => (*), :complex => ((a, b) -> (a + b) .* (a - b) .+ (-a .* b)),
)

const SLICE_OPS = Dict(
    :F_u => (
        (u, v, args) -> (
            -u[2:(end - 1), 2:(end - 1)] .*
            (v[2:(end - 1), 2:(end - 1)] .* v[2:(end - 1), 2:(end - 1)]) +
            args.f * (1 .- u[2:(end - 1), 2:(end - 1)])
        )
    ),
    :F_v => (
        (u, v, args) -> (
            u[2:(end - 1), 2:(end - 1)] .*
            (v[2:(end - 1), 2:(end - 1)] .* v[2:(end - 1), 2:(end - 1)]) -
            (args.f + args.k) * v[2:(end - 1), 2:(end - 1)]
        )
    ),
    :lap_u => (
        (u, _, args) -> (
            (
                u[3:end, 2:(end - 1)] .- 2*u[2:(end - 1), 2:(end - 1)] .+
                u[1:(end - 2), 2:(end - 1)]
            ) ./ args.dx^2 .+
            (
                u[2:(end - 1), 3:end] .- 2*u[2:(end - 1), 2:(end - 1)] .+
                u[2:(end - 1), 1:(end - 2)]
            ) ./ args.dx^2
        )
    ),
)

function test_scoping_rewrite_pipeline()
    utils = cuNumeric.ScopingUtils

    @testset "Syntax helpers" begin
        @test utils._assignment(:(x = y)) == (lhs=:x, rhs=:y)
        @test utils._broadcast_assignment(:(A[:] .= x)).rhs == :x
        @test utils._call(:(f(x, y))) == (f=:f, args=Any[:x, :y])
        @test utils._dotcall(:(f.(x, y))) == (f=:f, args=Any[:x, :y])
        @test utils._reference(:(A[i, j])) == (array=:A, indices=Any[:i, :j])
        @test utils._is_broadcast_syntax(:(x .* y))
        @test utils._is_scalar_expression(:(args.dx ^ 2))
        @test utils._is_scalar_expression(:(args.f + args.k))
        @test !utils._is_scalar_expression(:(A ^ 2))
        @test utils._replace_symbols(:(x .+ y), Dict(:x => :(a .* b))) ==
            :((a .* b) .+ y)
        @test isnothing(
            utils._assignment(quote
                x = y
            end),
        )
    end

    @testset "Inter-broadcast fusion" begin
        source = quote
            product = A .* B
            shifted = product .+ 2
            C[:, :] = shifted ./ 3
        end
        events = NamedTuple[]
        rewritten = cuNumeric.InterBroadcastFusion.rewrite_scope(
            source; on_rewrite=event -> push!(events, event)
        )
        stmts = utils._flatten_statements(rewritten)
        fused = sprint(Base.show_unquoted, only(stmts))
        @test !occursin("product", fused)
        @test !occursin("shifted", fused)
        @test occursin(".=", fused)

        io = IOBuffer()
        cuNumeric.InterBroadcastFusion.log_rewrite(only(events); io)
        log = String(take!(io))
        @test occursin("product =", log)
        @test occursin("shifted =", log)
        @test occursin("fused", log)

        untouched = quote
            C[:, :] = A .* B
        end
        empty!(events)
        rewritten = cuNumeric.InterBroadcastFusion.rewrite_scope(
            untouched; on_rewrite=event -> push!(events, event)
        )
        @test isempty(events)
        @test utils._strip_lines(rewritten) == utils._strip_lines(untouched)
    end

    @testset "Fusion-aware lifetime rewrite" begin
        source = quote
            C[2:(end - 1), 2:(end - 1)] .=
                A[2:(end - 1), 2:(end - 1)] .* B[2:(end - 1), 2:(end - 1)] .+ 2
        end

        cuNumeric.counter[] = 0
        try
            rewritten, assigned = cuNumeric.rewrite_broadcast_lifetimes(source)
            stmts = utils._flatten_statements(rewritten)

            @test assigned == Set([:tmp1, :tmp2, :tmp3])
            @test utils._assignment(stmts[1]).lhs == :tmp1
            @test utils._assignment(stmts[2]) ==
                (lhs=:tmp2, rhs=:(A[2:(end - 1), 2:(end - 1)]))
            @test utils._assignment(stmts[3]) ==
                (lhs=:tmp3, rhs=:(B[2:(end - 1), 2:(end - 1)]))
            @test !isnothing(utils._broadcast_assignment(stmts[4]))

            finalized = cuNumeric.insert_finalizers(rewritten, assigned)
            freed = Set{Symbol}()
            for stmt in utils._flatten_statements(finalized)
                argument = cuNumeric._delete_argument(stmt)
                isnothing(argument) || push!(freed, argument)
            end
            @test freed == assigned
        finally
            cuNumeric.counter[] = 0
        end
    end

    @testset "Scalar arithmetic stays inline" begin
        source = quote
            C .= A ./ args.dx^2 .+ (args.f + args.k)
        end

        cuNumeric.counter[] = 0
        try
            rewritten, assigned = cuNumeric.rewrite_broadcast_lifetimes(source)
            rendered = sprint(Base.show_unquoted, utils._strip_lines(rewritten))

            @test isempty(assigned)
            @test occursin("args.dx ^ 2", rendered)
            @test occursin("args.f + args.k", rendered)
        finally
            cuNumeric.counter[] = 0
        end
    end

    @testset "Eager lifetime rewrite" begin
        source = quote
            result = f(A[2:(end - 1), 2:(end - 1)])
            consume(result)
        end

        cuNumeric.counter[] = 0
        try
            rewritten, assigned = cuNumeric.rewrite_eager_lifetimes(source)
            stmts = utils._flatten_statements(rewritten)

            @test assigned == Set([:result, :tmp1, :tmp2, :tmp3])
            @test utils._assignment(stmts[1]) ==
                (lhs=:tmp1, rhs=:(A[2:(end - 1), 2:(end - 1)]))
            @test utils._call(utils._assignment(stmts[2]).rhs).f == :f
            @test utils._assignment(stmts[3]) == (lhs=:result, rhs=:tmp2)
            @test utils._call(utils._assignment(stmts[4]).rhs).f == :consume
        finally
            cuNumeric.counter[] = 0
        end
    end

    @testset "Multiple returned bindings" begin
        source = quote
            tmp1 = f(A)
            first_result = tmp1
            tmp2 = g(A)
            second_result = tmp2
            (first_result, second_result)
        end
        assigned = Set([:tmp1, :first_result, :tmp2, :second_result])
        finalized = cuNumeric.insert_finalizers(source, assigned)
        freed = Set(
            filter(
                !isnothing, map(cuNumeric._delete_argument, utils._flatten_statements(finalized))
            ),
        )

        @test isempty(freed)
    end

    @testset "Lexical lifetime scope" begin
        function hidden_binding()
            @analyze_lifetimes begin
                internal_result = 41
                nothing
            end
            return internal_result
        end

        function shadowed_binding()
            internal_result = :outer
            @analyze_lifetimes begin
                internal_result = :inner
                nothing
            end
            return internal_result
        end

        function hidden_destructured_bindings()
            @analyze_lifetimes begin
                internal_first, internal_second = (1, 2)
                nothing
            end
            return internal_first, internal_second
        end

        output = [0]
        returned = @analyze_lifetimes begin
            internal_result = 42
            output[1] = internal_result
            internal_result
        end

        @test_throws UndefVarError hidden_binding()
        @test_throws UndefVarError hidden_destructured_bindings()
        @test shadowed_binding() == :outer
        @test output == [42]
        @test returned == 42

        if cuNumeric.FUSE_BROADCAST_EXPRS
            function hidden_fused_binding(a, b, destination)
                @analyze_lifetimes begin
                    fused_result = a .* b
                    destination .= fused_result .+ 1
                end
                return fused_result
            end

            destination = zeros(Int, 2)
            @test_throws UndefVarError hidden_fused_binding(
                [2, 3], [4, 5], destination
            )
            @test destination == [9, 16]
        end
    end
end

function test_scoping_regressions(T, N)
    A = cuNumeric.ones(T, (N, N))
    B = cuNumeric.ones(T, (N, N))
    C = cuNumeric.zeros(T, (N, N))

    @testset "In-place assignment" begin
        @analyze_lifetimes begin
            result = A[1:end, :] .+ B[1:end, :]
            C .= result .* T(2.0)
        end
        # Test values: (1+1) * 2 = 4
        @test all(Array(C) .== T(4.0))
    end

    @testset "Macro as RHS" begin
        # Test values: (1+1)^2 = 4
        res = @analyze_lifetimes (A .+ B) .^ 2
        @test res isa cuNumeric.NDArray
        @test all(Array(res) .== T(4.0))
    end

    if cuNumeric.FUSE_BROADCAST_EXPRS
        @testset "Indexed fused assignment writes through NDArray slices" begin
            out = cuNumeric.zeros(T, (N + 2, N + 2))
            @analyze_lifetimes begin
                producer = A .* T(2)
                out[2:(end - 1), 2:(end - 1)] = producer .+ T(1)
            end
            expected = zeros(T, N + 2, N + 2)
            expected[2:(end - 1), 2:(end - 1)] .= T(3)
            @test Array(out) == expected
        end
    end
end

function run_all_ops(FT, N)
    results = Dict()

    # Regular binary/complex ops
    for (name, op) in OPS
        c_base, c_scoped = run_test(op, binary_scope(op), FT, N)
        results[name] = (c_base, c_scoped)
    end

    # Slice-heavy ops
    for (name, op) in SLICE_OPS
        c_base, c_scoped = run_slice_test(op, slice_scope(op), FT, N)
        results[name] = (c_base, c_scoped)
    end

    # Regression tests
    test_scoping_regressions(FT, N)

    return results
end
