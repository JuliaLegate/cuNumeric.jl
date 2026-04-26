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

    @testset "Chained intermediates freed" begin
        # t1 = A .+ A → 2.0; t2 = t1 .* t1 → 4.0
        res = @analyze_lifetimes begin
            t1 = A .+ A
            t1 .* t1
        end
        @test res isa cuNumeric.NDArray
        @test all(Array(res) .== T(4.0))
    end

    @testset "Slice as function argument" begin
        # A[2:N, 2:N] are all 1.0; adding two slices → 2.0
        res = @analyze_lifetimes A[2:N, 2:N] .+ A[2:N, 2:N]
        @test res isa cuNumeric.NDArray
        @test all(Array(res) .== T(2.0))
    end

    @testset "Boundary condition write-back" begin
        # Write column 1 of A (all 1.0) into column 1 of C (initially 0.0)
        @analyze_lifetimes C[:, 1] = A[:, 1]
        @test all(Array(C[:, 1]) .== T(1.0))
        # Reset for other tests
        C .= T(0.0)
    end

    @testset "Sequential slice-LHS setindex! block" begin
        D = cuNumeric.zeros(T, (N, N))
        @analyze_lifetimes begin
            D[:, 1] = A[:, end]
            D[:, end] = A[:, 1]
        end
        @test all(Array(D[:, 1]) .== T(1.0))
        @test all(Array(D[:, end]) .== T(1.0))
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
