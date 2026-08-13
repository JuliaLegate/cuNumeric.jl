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

#= Purpose of test: random
    -- dtype / shape of rand, randn, and in-place fills
    -- uniforms land in [0, 1) with mean 1/2 and variance 1/12
    -- normals match loc / scale moments (default and shifted)
    -- Monte-Carlo integral of exp(-x^2) recovers √π
    -- rand(a:b) is inclusive and a small discrete range has the right mean / var
    -- XORWOW / MRG32k3a / PHILOX4_32_10 all draw valid uniforms
=#

function _host(arr)
    return allowscalar() do
        return Array(arr)
    end
end

function _moments(arr)
    h = _host(arr)
    return mean(h), var(h)
end

@testset verbose = true "constructors" begin
    for T in (Float32, Float64)
        A = cuNumeric.rand(T, 8, 4)
        @test eltype(A) === T
        @test size(A) == (8, 4)
        @test all(0 .<= _host(A) .< 1)

        B = cuNumeric.zeros(T, 5, 5)
        cuNumeric.rand!(B)
        @test all(0 .<= _host(B) .< 1)

        N = cuNumeric.randn(T, 32)
        @test eltype(N) === T
        @test size(N) == (32,)

        C = cuNumeric.zeros(T, 16)
        Random.randn!(C)
        @test eltype(C) === T
    end

    R = cuNumeric.rand(3, 2)
    @test eltype(R) === Float32
    @test size(R) == (3, 2)

    @test_throws MethodError cuNumeric.rand(Bool, 4)
    @test_throws MethodError cuNumeric.randn(Int32, 4)
end

@testset verbose = true "uniform moments" begin
    n = 65_536
    for T in (Float32, Float64)
        μ, v = _moments(cuNumeric.rand(T, n))
        @test abs(μ - 0.5) < 0.03
        @test abs(v - 1 / 12) < 0.01

        μ2, v2 = _moments(cuNumeric.rand(T, 128, 128))
        @test abs(μ2 - 0.5) < 0.03
        @test abs(v2 - 1 / 12) < 0.01
    end
end

@testset verbose = true "normal moments" begin
    n = 65_536
    for T in (Float32, Float64)
        μ, v = _moments(cuNumeric.randn(T, n))
        @test abs(μ) < 0.05
        @test abs(v - 1) < 0.08

        g = cuNumeric.default_rng()
        A = cuNumeric.zeros(T, n)
        randn!(g, A; loc=T(3), scale=T(2))
        μs, vs = _moments(A)
        @test abs(μs - 3) < 0.1
        @test abs(vs - 4) < 0.3
    end
end

@testset verbose = true "monte carlo" begin
    # ∫_{-∞}^{∞} exp(-x^2) dx = √π, truncated to [-10, 10] as in the docs example.
    n = 131_072
    for T in (Float32, Float64)
        xmax = T(10)
        Ω = T(2) * xmax
        samples = Ω .* cuNumeric.rand(T, n) .- xmax
        integrand = (x) -> @. exp(-x^2)
        estimate = unwrap((Ω / n) * sum(integrand(samples)))
        @test isapprox(estimate, T(sqrt(π)); atol=T(0.08))
    end

    # Area of the unit disk via darts in [0, 1]^2 → π.
    n = 131_072
    for T in (Float32, Float64)
        x = _host(cuNumeric.rand(T, n))
        y = _host(cuNumeric.rand(T, n))
        π_estimate = 4 * mean(x .^ 2 .+ y .^ 2 .< one(T))
        @test isapprox(π_estimate, T(π); atol=T(0.05))
    end
end

@testset verbose = true "integers" begin
    A = cuNumeric.rand(0:9, 64)
    @test eltype(A) === Int
    @test all(0 .<= _host(A) .<= 9)

    B = cuNumeric.rand(Int32(-4):Int32(4), 8, 8)
    @test eltype(B) === Int32
    @test size(B) == (8, 8)
    @test all(-4 .<= _host(B) .<= 4)

    C = cuNumeric.rand(Int16, 32)
    @test eltype(C) === Int16
    @test all(typemin(Int16) .<= _host(C) .< typemax(Int16))

    D = cuNumeric.zeros(Int32, 16)
    cuNumeric.rand!(D)
    @test all(typemin(Int32) .<= _host(D) .< typemax(Int32))

    # singleton range is constant
    Z = _host(cuNumeric.rand(0:0, 256))
    @test all(==(0), Z)
    F = _host(cuNumeric.rand(Int32(5):Int32(5), 128))
    @test all(==(5), F)

    # discrete uniform on 0:9: mean 4.5, var (n^2-1)/12 with n=10
    n = 65_536
    h = Float64.(_host(cuNumeric.rand(0:9, n)))
    @test abs(mean(h) - 4.5) < 0.08
    @test abs(var(h) - (100 - 1) / 12) < 0.2

    @test_throws ArgumentError cuNumeric.rand(1:0, 4)
end

@testset verbose = true "default_rng" begin
    g = cuNumeric.default_rng()
    @test g isa cuNumeric.Generator
    A = cuNumeric.random(g, Float32, (4, 4))
    @test eltype(A) === Float32
    @test size(A) == (4, 4)
    @test all(0 .<= _host(A) .< 1)

    g2 = cuNumeric.default_rng(1234)
    @test g2 isa cuNumeric.Generator
    @test g2.bit_generator isa cuNumeric.XORWOW
    @test g2.bit_generator.seed == UInt64(1234)
    B = cuNumeric.randn(g2, Float64, (32,))
    @test eltype(B) === Float64

    s1 = cuNumeric.get_static_generator()
    s2 = cuNumeric.get_static_generator()
    @test s1 === s2
    @test s1 !== g2
end

@testset verbose = true "bitgenerators" begin
    n = 16_384
    for B in (cuNumeric.XORWOW, cuNumeric.MRG32k3a, cuNumeric.PHILOX4_32_10)
        @testset "$(B)" begin
            g = cuNumeric.default_rng(B, 42)
            @test g isa cuNumeric.Generator{B}
            @test g.bit_generator isa B
            @test g.bit_generator.seed == UInt64(42)

            U = cuNumeric.random(g, Float32, (n,))
            @test eltype(U) === Float32
            Uh = _host(U)
            @test all(0 .<= Uh .< 1)
            @test abs(mean(Uh) - 0.5) < 0.05
            @test abs(var(Uh) - 1 / 12) < 0.02

            Nrm = cuNumeric.randn(g, Float64, (n,))
            μ, v = _moments(Nrm)
            @test abs(μ) < 0.08
            @test abs(v - 1) < 0.12
        end
    end
end
