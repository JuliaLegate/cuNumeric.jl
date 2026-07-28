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

#= Broadcast fusion arg buffer tests.
 * Tests every combination of array and scalar argument positions
 * to verify the arg_map protocol correctly reconstructs the CUDA
 * kernel arg buffer.
 *
 * Pattern naming convention:
 *   A = NDArray input
 *   S = scalar input
 *   Position = left-to-right in the broadcast expression
 *   "same" = same array appears multiple times (deduplication test)
=#

_broadcast_fusion_user_add(x, y) = x + y

function test_broadcast_fusion(; T=Float32, N=100, atol=1e-5, rtol=1e-5)
    # Create test arrays with known non-zero values
    julia_a = rand(T, N)
    julia_b = rand(T, N)
    julia_c = rand(T, N)

    a = @allowscalar NDArray(julia_a)
    b = @allowscalar NDArray(julia_b)
    c = @allowscalar NDArray(julia_c)

    s1 = T(2.5)
    s2 = T(1.0)
    s3 = T(0.5)

    # two different arrays
    @testset "A + B (two different arrays)" begin
        expected = julia_a .+ julia_b
        result = a .+ b
        @allowscalar @test cuNumeric.compare(expected, result, atol, rtol)
    end

    # same array, deduplication
    @testset "A + A (same array twice)" begin
        expected = julia_a .+ julia_a
        result = a .+ a
        @allowscalar @test cuNumeric.compare(expected, result, atol, rtol)
    end

    # array then scalar
    @testset "A + scalar (array first)" begin
        expected = julia_a .+ s1
        result = a .+ s1
        @allowscalar @test cuNumeric.compare(expected, result, atol, rtol)
    end

    # scalar then array
    @testset "scalar + A (scalar first)" begin
        expected = s1 .+ julia_a
        result = s1 .+ a
        @allowscalar @test cuNumeric.compare(expected, result, atol, rtol)
    end

    # array, two scalars, fused
    @testset "A * scalar - scalar (fused)" begin
        expected = julia_a .* s1 .- s2
        result = a .* s1 .- s2
        @allowscalar @test cuNumeric.compare(expected, result, atol, rtol)
    end

    # scalar-array-scalar
    @testset "scalar * A + scalar" begin
        expected = s1 .* julia_a .+ s2
        result = s1 .* a .+ s2
        @allowscalar @test cuNumeric.compare(expected, result, atol, rtol)
    end

    # two arrays then scalar
    @testset "A + B + scalar" begin
        expected = julia_a .+ julia_b .+ s1
        result = a .+ b .+ s1
        @allowscalar @test cuNumeric.compare(expected, result, atol, rtol)
    end

    # scalar then two arrays
    @testset "scalar + A + B" begin
        expected = s1 .+ julia_a .+ julia_b
        result = s1 .+ a .+ b
        @allowscalar @test cuNumeric.compare(expected, result, atol, rtol)
    end

    # same array twice + scalar (dedup)
    @testset "A + A + scalar (dedup + scalar)" begin
        expected = julia_a .+ julia_a .+ s1
        result = a .+ a .+ s1
        @allowscalar @test cuNumeric.compare(expected, result, atol, rtol)
    end

    # three different arrays
    @testset "A + B + C (three arrays)" begin
        expected = julia_a .+ julia_b .+ julia_c
        result = a .+ b .+ c
        @allowscalar @test cuNumeric.compare(expected, result, atol, rtol)
    end

    # same array three times (triple dedup)
    @testset "A + A + A (triple dedup)" begin
        expected = julia_a .+ julia_a .+ julia_a
        result = a .+ a .+ a
        @allowscalar @test cuNumeric.compare(expected, result, atol, rtol)
    end

    # two scalars then array
    @testset "scalar * scalar + A" begin
        expected = s1 .* s2 .+ julia_a
        result = s1 .* s2 .+ a
        @allowscalar @test cuNumeric.compare(expected, result, atol, rtol)
    end

    # multiply, two arrays (different PTX kernel name collision test)
    @testset "A * B (multiply)" begin
        expected = julia_a .* julia_b
        result = a .* b
        @allowscalar @test cuNumeric.compare(expected, result, atol, rtol)
    end

    # same array squared, dedup
    @testset "A * A (self multiply, dedup)" begin
        expected = julia_a .* julia_a
        result = a .* a
        @allowscalar @test cuNumeric.compare(expected, result, atol, rtol)
    end

    # subtraction, order matters
    @testset "A - B (subtraction)" begin
        expected = julia_a .- julia_b
        result = a .- b
        @allowscalar @test cuNumeric.compare(expected, result, atol, rtol)
    end

    # scalar minus array
    @testset "scalar - A" begin
        expected = s1 .- julia_a
        result = s1 .- a
        @allowscalar @test cuNumeric.compare(expected, result, atol, rtol)
    end

    # three arrays, mixed ops
    @testset "A * B + C (three arrays, mixed ops)" begin
        expected = julia_a .* julia_b .+ julia_c
        result = a .* b .+ c
        @allowscalar @test cuNumeric.compare(expected, result, atol, rtol)
    end

    # two arrays fused, then scaled
    @testset "(A + B) * scalar" begin
        expected = (julia_a .+ julia_b) .* s1
        result = (a .+ b) .* s1
        @allowscalar @test cuNumeric.compare(expected, result, atol, rtol)
    end

    if cuNumeric.FUSE_BROADCAST_EXPRS
        @testset "z .= scalar * f.(A, B)" begin
            expected = T(2.0) .* (julia_a .+ julia_b)
            z = cuNumeric.zeros(T, (N,))
            @analyze_lifetimes begin
                z .= T(2.0) .* _broadcast_fusion_user_add.(a, b)
            end
            @allowscalar @test cuNumeric.compare(expected, z, atol, rtol)
        end
    end

    # scalar-array pairs
    @testset "scalar * A + scalar * B" begin
        expected = s1 .* julia_a .+ s2 .* julia_b
        result = s1 .* a .+ s2 .* b
        @allowscalar @test cuNumeric.compare(expected, result, atol, rtol)
    end

    # same array subtracted, should be all zeros
    @testset "A - A (same array, expect zeros)" begin
        expected = julia_a .- julia_a
        result = a .- a
        @allowscalar @test cuNumeric.compare(expected, result, atol, rtol)
    end
end

#= Edge cases for same-shaped broadcast fusion.
 * Complements `test_broadcast_fusion` with size extremes, 2D same-shape,
 * fusion gating for shape mismatch, 0-d fallback, and dest/input aliasing.
=#
function test_broadcast_fusion_edge_cases(; T=Float32, atol=1e-5, rtol=1e-5)
    s1 = T(2.5)
    s2 = T(1.0)

    @testset "very small 1D (N=1)" begin
        ja = T[1.5]
        jb = T[2.25]
        a = @allowscalar NDArray(ja)
        b = @allowscalar NDArray(jb)
        result = a .+ b .* s1 .- s2
        @allowscalar @test cuNumeric.compare(ja .+ jb .* s1 .- s2, result, atol, rtol)
    end

    @testset "very small 1D (N=2)" begin
        ja = T[1.5, 3.0]
        jb = T[2.5, 4.0]
        a = @allowscalar NDArray(ja)
        b = @allowscalar NDArray(jb)
        result = a .+ b
        @allowscalar @test cuNumeric.compare(ja .+ jb, result, atol, rtol)
        result = s1 .* a .- b
        @allowscalar @test cuNumeric.compare(s1 .* ja .- jb, result, atol, rtol)
    end

    @testset "empty / zero-size 1D" begin
        # NDArray supports size (0,). `_copyto!` short-circuits on isempty
        # before fusion, so this only checks the empty path does not crash.
        e = cuNumeric.zeros(T, 0)
        result = e .+ e
        @test isempty(result)
        @test size(result) == (0,)
        result = e .+ s1
        @test isempty(result)
        @test size(result) == (0,)
    end

    @testset "empty / zero-size 2D" begin
        e = cuNumeric.zeros(T, (0, 3))
        result = e .+ e
        @test isempty(result)
        @test size(result) == (0, 3)
    end

    @testset "large-ish 1D same-shape fused" begin
        N = 10_000
        ja = rand(T, N)
        jb = rand(T, N)
        a = @allowscalar NDArray(ja)
        b = @allowscalar NDArray(jb)
        result = a .+ b .* s1 .- s2
        @allowscalar @test cuNumeric.compare(ja .+ jb .* s1 .- s2, result, atol, rtol)
        # Gate: same-shape leaves should be fusible.
        dest = cuNumeric.zeros(T, N)
        bc = Base.Broadcast.instantiate(Base.broadcasted(+, a, b))
        @test cuNumeric.can_fuse_linear_broadcast(dest, bc)
    end

    @testset "large-ish 2D same-shape fused" begin
        M, N = 128, 256
        ja = rand(T, M, N)
        jb = rand(T, M, N)
        a = @allowscalar NDArray(ja)
        b = @allowscalar NDArray(jb)
        result = a .+ b .* s1
        @allowscalar @test cuNumeric.compare(ja .+ jb .* s1, result, atol, rtol)
        result = a .+ a .* b .- s2
        @allowscalar @test cuNumeric.compare(ja .+ ja .* jb .- s2, result, atol, rtol)
        dest = cuNumeric.zeros(T, M, N)
        bc = Base.Broadcast.instantiate(Base.broadcasted(+, a, b))
        @test cuNumeric.can_fuse_linear_broadcast(dest, bc)
    end

    @testset "2D Cartesian launch shapes" begin
        for (M, N) in ((1, 513), (513, 1), (37, 513))
            ja = rand(T, M, N)
            jb = rand(T, M, N)
            a = @allowscalar NDArray(ja)
            b = @allowscalar NDArray(jb)
            result = a .+ b .* s1
            @allowscalar @test cuNumeric.compare(ja .+ jb .* s1, result, atol, rtol)
        end
    end

    @testset "scalar gaps: A .^ 2 and scalar*A*scalar" begin
        N = 64
        ja = rand(T, N)
        a = @allowscalar NDArray(ja)
        # Literal power uses RefValue{Val} / static-arg lowering.
        result = a .^ 2
        @allowscalar @test cuNumeric.compare(ja .^ 2, result, atol, rtol)
        result = s1 .* a .* s2
        @allowscalar @test cuNumeric.compare(s1 .* ja .* s2, result, atol, rtol)
    end

    # Gray-Scott-style slice stencils: strided views must fuse correctly via
    # CuStridedDeviceArray (packed Legate element strides).
    @testset "fused slice stencils (X/Y and slice dest)" begin
        N = 32
        ja = rand(T, N, N)
        u = @allowscalar NDArray(ja)
        two = T(2)

        # X-shifted (vary first index)
        result_x =
            u[3:end, 2:(end - 1)] .- 2 .* u[2:(end - 1), 2:(end - 1)] .+
            u[1:(end - 2), 2:(end - 1)]
        expected_x =
            ja[3:end, 2:(end - 1)] .- two .* ja[2:(end - 1), 2:(end - 1)] .+
            ja[1:(end - 2), 2:(end - 1)]
        @allowscalar @test cuNumeric.compare(expected_x, result_x, atol, rtol)

        dest_x = similar(result_x)
        bc_x = Base.Broadcast.instantiate(
            Base.broadcasted(
                +,
                Base.broadcasted(
                    -,
                    u[3:end, 2:(end - 1)],
                    Base.broadcasted(*, 2, u[2:(end - 1), 2:(end - 1)]),
                ),
                u[1:(end - 2), 2:(end - 1)],
            ),
        )
        @test cuNumeric.can_fuse_linear_broadcast(dest_x, bc_x)

        # Y-shifted (vary second index) — previously failed under dense packing
        result_y =
            u[2:(end - 1), 3:end] .- 2 .* u[2:(end - 1), 2:(end - 1)] .+
            u[2:(end - 1), 1:(end - 2)]
        expected_y =
            ja[2:(end - 1), 3:end] .- two .* ja[2:(end - 1), 2:(end - 1)] .+
            ja[2:(end - 1), 1:(end - 2)]
        @allowscalar @test cuNumeric.compare(expected_y, result_y, atol, rtol)

        # Assign into a slice destination
        out = @allowscalar NDArray(zeros(T, N, N))
        out[2:(end - 1), 2:(end - 1)] = result_x .+ result_y
        expected_out = zeros(T, N, N)
        expected_out[2:(end - 1), 2:(end - 1)] = expected_x .+ expected_y
        @allowscalar @test cuNumeric.compare(expected_out, out, atol, rtol)
    end

    @testset "fused/unfused scalar promotion parity" begin
        N = 32
        ja = rand(T, N)
        a = @allowscalar NDArray(ja)
        dest = cuNumeric.zeros(T, N)

        # Bare Int64: fusible, host-promoted to T, same result as unfused.
        bc_i64 = Base.Broadcast.instantiate(Base.broadcasted(*, 2, a))
        @test cuNumeric.can_fuse_linear_broadcast(dest, bc_i64)
        result = 2 .* a
        @allowscalar @test cuNumeric.compare(T(2) .* ja, result, atol, rtol)
        @allowscalar @test cuNumeric.compare(
            result,
            cuNumeric.unravel_broadcast_tree(bc_i64),
            atol,
            rtol,
        )

        # Wider Float64 literal: fused and unfused throw the same promotion error.
        if T === Float32
            bc_f64 = Base.Broadcast.instantiate(Base.broadcasted(+, a, 1.0))
            err_fused = try
                a .+ 1.0
                nothing
            catch e
                sprint(showerror, e)
            end
            err_unfused = try
                cuNumeric.unravel_broadcast_tree(bc_f64)
                nothing
            catch e
                sprint(showerror, e)
            end
            @test err_fused !== nothing
            @test err_unfused !== nothing
            @test err_fused == err_unfused
            @test occursin("Implicit promotion", err_fused)
            @test_throws "Implicit promotion" a .+ 1.0
        end
    end

    @testset "shape-mismatched broadcast refuses fusion" begin
        # Linear fusion requires every NDArray leaf to match dest shape.
        # Matrix .+ vector must fall back to the unfused path.
        #
        # NOTE: the unfused matrix.+vector path currently disagrees with Julia
        # broadcasting semantics; do not assert equality with `ja .+ jv` here.
        M, N = 64, 32
        ja = rand(T, M, N)
        jv = rand(T, M)
        a = @allowscalar NDArray(ja)
        v = @allowscalar NDArray(jv)
        dest = cuNumeric.zeros(T, M, N)
        bc = Base.Broadcast.instantiate(Base.broadcasted(+, a, v))
        @test !cuNumeric.can_fuse_linear_broadcast(dest, bc)

        # Unfused fallback should not crash (correctness vs Julia is known-wrong).
        result = a .+ v
        @test size(result) == (M, N)
    end

    @testset "0-d scalar NDArray (fusion refused, unfused ok)" begin
        # RunPTXBroadcastTask only supports dims in [1, 6]; can_fuse refuses
        # 0-d so `_copyto!` falls back to unfused.
        #
        # NOTE: `z1 .+ z2` still errors after a successful `copyto!` because
        # `Broadcast.copy` for NDArrayStyle{0} unwraps with
        # `dest[CartesianIndex()]`, which is not implemented. Test via
        # `copyto!` into an explicit 0-d dest instead.
        z1 = @allowscalar NDArray(T(2))
        z2 = @allowscalar NDArray(T(3))
        @test ndims(z1) == 0
        dest = cuNumeric.zeros(T)
        bc = Base.Broadcast.instantiate(Base.broadcasted(+, z1, z2))
        @test !cuNumeric.can_fuse_linear_broadcast(dest, bc)
        copyto!(dest, bc)
        @allowscalar @test dest[] == T(5)

        dest2 = cuNumeric.zeros(T)
        bc2 = Base.Broadcast.instantiate(Base.broadcasted(+, Base.broadcasted(*, z1, s1), z2))
        @test !cuNumeric.can_fuse_linear_broadcast(dest2, bc2)
        copyto!(dest2, bc2)
        @allowscalar @test dest2[] ≈ T(2) * s1 + T(3) atol = atol rtol = rtol
    end

    @testset "dest aliases an input" begin
        N = 128
        ja = rand(T, N)
        jb = rand(T, N)
        a = @allowscalar NDArray(copy(ja))
        b = @allowscalar NDArray(jb)
        a .+= b
        @allowscalar @test cuNumeric.compare(ja .+ jb, a, atol, rtol)

        a2 = @allowscalar NDArray(copy(ja))
        a2 .= a2 .* s1 .+ b
        @allowscalar @test cuNumeric.compare(ja .* s1 .+ jb, a2, atol, rtol)
    end
end

#= Broadcast fusion PTX compilation cache.
 * Verifies `_BCAST_PTX_CACHE` grows on first fused launch of a signature and
 * is reused (no new entry) on a second launch of the same signature.
 * Gated on `FUSE_BROADCAST_EXPRS` + `HAS_CUDA`; skips otherwise.
 * With `FUSE_BROADCAST_MIN_OPS > 1`, single-op exprs are unfused — tests
 * should set min ops to 1 (LocalPreferences / ENV) to exercise the cache.
=#
function test_broadcast_fusion_ptx_cache(; T=Float32, N=64)
    if !(cuNumeric.FUSE_BROADCAST_EXPRS && cuNumeric.HAS_CUDA)
        @info "Skipping PTX cache tests (need FUSE_BROADCAST_EXPRS && HAS_CUDA)"
        return nothing
    end
    if cuNumeric.FUSE_BROADCAST_MIN_OPS > 1
        @info "Skipping PTX cache tests (need FUSE_BROADCAST_MIN_OPS <= 1 to fuse single-op exprs)"
        return nothing
    end

    cache = cuNumeric._BCAST_PTX_CACHE
    cache_lock = cuNumeric._BCAST_PTX_CACHE_LOCK
    cache_len() =
        lock(cache_lock) do
            return length(cache)
        end
    clear_cache!() =
        lock(cache_lock) do
            empty!(cache)
            return nothing
        end

    @testset "PTX cache hit / miss" begin
        clear_cache!()
        @test cache_len() == 0

        a = @allowscalar NDArray(rand(T, N))
        b = @allowscalar NDArray(rand(T, N))

        # First fused launch of a signature should compile and cache.
        _ = a .+ b
        n1 = cache_len()
        @test n1 >= 1

        # Same signature again should hit the cache (no new entry).
        _ = a .+ b
        @test cache_len() == n1

        # Different op should miss and add a new entry.
        _ = a .* b
        n2 = cache_len()
        @test n2 > n1

        # Same multiply signature again should hit.
        _ = a .* b
        @test cache_len() == n2

        # Different element type should miss and add another entry.
        a64 = @allowscalar NDArray(rand(Float64, N))
        b64 = @allowscalar NDArray(rand(Float64, N))
        _ = a64 .+ b64
        n3 = cache_len()
        @test n3 > n2

        _ = a64 .+ b64
        @test cache_len() == n3

        # Nested fused expression should miss, then hit on re-run.
        _ = a .+ b .* a
        n4 = cache_len()
        @test n4 > n3

        _ = a .+ b .* a
        @test cache_len() == n4
    end
end
