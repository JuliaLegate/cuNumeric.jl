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

# Coverage for src/ndarray/diagonal.jl: diag/_eye/trace, Diagonal, UniformScaling.

const DIAGONAL_NUMERIC_TYPES = Base.uniontypes(cuNumeric.SUPPORTED_NUMERIC_TYPES)
const DIAGONAL_ARRAY_TYPES = Base.uniontypes(cuNumeric.SUPPORTED_ARRAY_TYPES)
const DIAGONAL_FLOAT_TYPES = Base.uniontypes(cuNumeric.SUPPORTED_FLOAT_TYPES)
const DIAGONAL_COMPLEX_TYPES = Base.uniontypes(cuNumeric.SUPPORTED_COMPLEX_TYPES)

_nonzero_diag(::Type{T}, n) where {T<:AbstractFloat} = abs.(my_rand(T, n)) .+ one(T)
function _nonzero_diag(::Type{T}, n) where {T<:Complex}
    return Complex.(abs.(real(my_rand(T, n))) .+ one(real(T)), zero(real(T)))
end
_nonzero_diag(::Type{T}, n) where {T<:Integer} = T.(collect(2:(n + 1)))
_nonzero_diag(::Type{Bool}, n) = fill(true, n)

function _host_diag_compare(ref, out, ::Type{T}) where {T}
    allowscalar() do
        @test cuNumeric.compare(ref, out, atol(T), rtol(T))
    end
end

function _host_scalar_compare(ref, out::NDArray{<:Any,0}, ::Type{T}) where {T}
    allowscalar() do
        @test ref ≈ out[] atol=atol(T) rtol=rtol(T)
    end
end

function _host_bool_compare(ref::Bool, out::NDArray{Bool,0})
    allowscalar() do
        @test out[] == ref
    end
end

function _host_matrix_compare(ref::AbstractMatrix, out::NDArray, ::Type{T}) where {T}
    allowscalar() do
        @test cuNumeric.compare(ref, out, atol(T), rtol(T))
    end
end

###### diag / identity / trace ######

@testset "diag" begin
    @testset verbose=true for T in DIAGONAL_NUMERIC_TYPES
        A = my_rand(T, 6, 5)
        nda = NDArray(A)
        @testset "k=$k" for k in (-2, 0, 2)
            ref = diag(A, k)
            _host_diag_compare(ref, cuNumeric.diag(nda; k=k), T)
            _host_diag_compare(ref, LinearAlgebra.diag(nda, k), T)
        end
    end
end

@testset "identity via I / _eye" begin
    @testset verbose=true for T in DIAGONAL_NUMERIC_TYPES
        n = 4
        ref = Matrix{T}(I, n, n)
        _host_matrix_compare(ref, NDArray{T}(I, n, n), T)
        _host_matrix_compare(ref, cuNumeric._eye(T, n), T)
    end
    # Untyped NDArray(I, ...) uses Bool; typed / _eye default to Float32 densify
    ref_bool = Matrix{Bool}(I, 3, 3)
    _host_matrix_compare(ref_bool, NDArray(I, 3, 3), Bool)
    ref = Matrix{Float32}(I, 3, 3)
    _host_matrix_compare(ref, NDArray{Float32}(I, 3, 3), Float32)
    _host_matrix_compare(ref, cuNumeric._eye(3), Float32)
end

@testset "trace" begin
    @testset verbose=true for T in DIAGONAL_NUMERIC_TYPES
        A = my_rand(T, 5, 5)
        nda = NDArray(A)
        @testset "offset=$k" for k in (-2, -1, 0, 1, 2)
            ref = sum(diag(A, k))
            out = cuNumeric.trace(nda; offset=k)
            @test out isa NDArray{<:Any,0}
            _host_scalar_compare(ref, out, eltype(ref))
        end
    end
end

###### Diagonal constructors / densify / show ######

@testset "Diagonal constructors" begin
    @testset verbose=true for T in DIAGONAL_NUMERIC_TYPES
        d = my_rand(T, 4)
        v = NDArray(d)
        D = Diagonal(v)
        @test D isa Diagonal{T,<:NDArray{T,1}}
        @test D.diag === v
        allowscalar() do
            @test cuNumeric.compare(d, D.diag, atol(T), rtol(T))
            @test Matrix(D) ≈ Matrix(Diagonal(d)) atol=atol(T) rtol=rtol(T)
            @test Matrix{T}(D) ≈ Matrix(Diagonal(d)) atol=atol(T) rtol=rtol(T)
        end

        A = my_rand(T, 4, 4)
        D2 = Diagonal(NDArray(A))
        allowscalar() do
            @test cuNumeric.compare(diag(A), D2.diag, atol(T), rtol(T))
        end
    end
end

@testset "Diagonal show" begin
    D = Diagonal(NDArray(Float32[1, 2, 3]))
    s = sprint(show, D)
    @test occursin("1.0", s) && occursin("2.0", s)
    plain = sprint(show, MIME"text/plain"(), D)
    # Summary must reflect the real Diagonal{T,<:NDArray} type, not Vector.
    @test occursin("NDArray", plain)
    @test occursin(string(typeof(D)), plain)
    @test occursin("1.0", plain)
    # Match Base Diagonal formatting (⋅ off-diagonals), not a dense Matrix dump.
    @test occursin("⋅", plain)
    dense = sprint(show, MIME"text/plain"(), Matrix(D))
    @test plain != dense
end

###### Diagonal operators ######

@testset "Diagonal * Diagonal / ± / scalar" begin
    @testset verbose=true for T in DIAGONAL_NUMERIC_TYPES
        d = my_rand(T, 3)
        Dh = Diagonal(d)
        D = Diagonal(NDArray(d))
        allowscalar() do
            @test Matrix(D * D) ≈ Matrix(Dh * Dh) atol=atol(T) rtol=rtol(T)
            @test Matrix(D + D) ≈ Matrix(Dh + Dh) atol=atol(T) rtol=rtol(T)
            @test Matrix(D - D) ≈ Matrix(Dh - Dh) atol=atol(T) rtol=rtol(T)
            @test Matrix(T(3) * D) ≈ Matrix(T(3) * Dh) atol=atol(T) rtol=rtol(T)
            @test Matrix(D * T(3)) ≈ Matrix(Dh * T(3)) atol=atol(T) rtol=rtol(T)
        end
    end
end

@testset "Diagonal broadcast on diag" begin
    @testset verbose=true for T in DIAGONAL_NUMERIC_TYPES
        d = my_rand(T, 5)
        c = T(5)
        Dh = Diagonal(d)
        D = Diagonal(NDArray(copy(d)))

        ref = collect((Dh .* c).diag)

        # Out-of-place: stays Diagonal with NDArray diag
        D2 = D .* c
        @test D2 isa Diagonal{T,<:NDArray{T,1}}
        @test D2.diag isa NDArray{T,1}
        _host_diag_compare(ref, D2.diag, T)

        # In-place fused assign
        D3 = Diagonal(NDArray(copy(d)))
        D3 .= D3 .* c
        @test D3 isa Diagonal{T,<:NDArray{T,1}}
        _host_diag_compare(ref, D3.diag, T)

        # In-place .*=
        D4 = Diagonal(NDArray(copy(d)))
        D4 .*= c
        @test D4 isa Diagonal{T,<:NDArray{T,1}}
        _host_diag_compare(ref, D4.diag, T)

        # Zero-preserving Diagonal .+ Diagonal (matches Base structure)
        ref_add = collect((Dh .+ Dh).diag)
        D_add = D .+ D
        @test D_add isa Diagonal{T,<:NDArray{T,1}}
        _host_diag_compare(ref_add, D_add.diag, T)

        D_add2 = Diagonal(NDArray(copy(d)))
        D_add2 .+= D_add2
        @test D_add2 isa Diagonal{T,<:NDArray{T,1}}
        _host_diag_compare(ref_add, D_add2.diag, T)

        D_add3 = Diagonal(NDArray(copy(d)))
        D_add3 .= D_add3 .+ D_add3
        @test D_add3 isa Diagonal{T,<:NDArray{T,1}}
        _host_diag_compare(ref_add, D_add3.diag, T)
    end

    # Exact repro from the bug report
    a = Diagonal(cuNumeric.ones(Int32, 5))
    a .*= Int32(5)
    @test a isa Diagonal{Int32,<:NDArray{Int32,1}}
    @test Array(a.diag) == fill(Int32(5), 5)
end

@testset "scalar indexing message: LinearAlgebra / Base vs plain" begin
    # Unsupported LA/Base fallbacks enrich; intentional user scalar indexing stays plain.
    allowscalar(false)

    function _assert_enriched_fallback(msg, modfunc)
        @test occursin("`$modfunc` fell back to an AbstractArray implementation", msg)
        @test occursin("which scalar-indexed an `NDArray`", msg)
        @test occursin("path is probably not implemented yet for `NDArray`", msg)
        @test occursin("allowscalar", msg)
        @test occursin("@allowscalar", msg)
        @test occursin("might allow this function to work slowly", msg)
        @test occursin("it has not been tested", msg)
        # Enriched path replaces the generic iterating-method lead and omits the plain header.
        @test !occursin("typically caused by calling an iterating implementation", msg)
        @test !occursin("Scalar indexing is disallowed", msg)
        @test !occursin("If you want to allow scalar iteration", msg)
        @test !occursin("triggered via", msg)
        @test startswith(lstrip(msg), "`$modfunc`")
    end

    err = @test_throws ErrorException cholesky(Diagonal(NDArray(Float32[2, 3, 4])))
    msg = sprint(showerror, err.value)
    _assert_enriched_fallback(msg, "LinearAlgebra.cholesky")

    # sortperm-based LA path must blame svd, not the inner Base.lt helper.
    err_svd = @test_throws ErrorException svd(Diagonal(NDArray(Float32[2, 3, 4])))
    msg_svd = sprint(showerror, err_svd.value)
    _assert_enriched_fallback(msg_svd, "LinearAlgebra.svd")
    @test !occursin("Base.lt", msg_svd)

    # Base AbstractArray fallback (unique) should enrich with Base.<func>.
    err_base = @test_throws ErrorException unique(NDArray(Float32[1, 2, 1]))
    msg_base = sprint(showerror, err_base.value)
    _assert_enriched_fallback(msg_base, "Base.unique")

    # Call through a Main function so the first non-cuNumeric/Core frame is user code,
    # not Base.include_string / Base.IncludeInto (Julia 1.12+) / client frames from
    # `include`ing this test file.
    function _plain_scalar_index_probe()
        a = NDArray(Float32[1, 2, 3])
        return a[1]
    end
    err_plain = try
        _plain_scalar_index_probe()
        nothing
    catch e
        e
    end
    @test err_plain isa ErrorException
    msg_plain = sprint(showerror, err_plain)
    @test occursin("Scalar indexing is disallowed", msg_plain)
    @test occursin("typically caused by calling an iterating implementation", msg_plain)
    @test occursin("If you want to allow scalar iteration", msg_plain)
    @test !occursin("triggered via", msg_plain)
    @test !occursin("fell back to an AbstractArray", msg_plain)
    @test !occursin("probably not implemented yet", msg_plain)
    @test !occursin("might allow this function to work slowly", msg_plain)
    @test !occursin("it has not been tested", msg_plain)
end

@testset "NDArray iszero / isone" begin
    z = cuNumeric.zeros(Float32, 3)
    o = cuNumeric.ones(Float32, 3)
    _host_bool_compare(true, iszero(z))
    _host_bool_compare(false, iszero(o))
    I = one(cuNumeric.zeros(Float32, 3, 3))
    _host_bool_compare(true, isone(I))
    _host_bool_compare(false, isone(cuNumeric.ones(Float32, 3, 3)))
    _host_bool_compare(false, isone(cuNumeric.zeros(Float32, 3, 3)))
    _host_bool_compare(false, isone(cuNumeric.ones(Float32, 2, 3)))
end

@testset "Diagonal densifying broadcast" begin
    D = Diagonal(NDArray(Float32[1, 2, 3]))
    expected = Float32[2 1 1; 1 3 1; 1 1 4]

    # Out-of-place densifies to dense NDArray (Base densifies to Matrix).
    R = D .+ Float32(1)
    @test R isa NDArray{Float32,2}
    @test Array(R) == expected

    A = cuNumeric.ones(Float32, 3, 3)
    R2 = D .+ A
    @test R2 isa NDArray{Float32,2}
    @test Array(R2) == expected
    R3 = A .+ D
    @test R3 isa NDArray{Float32,2}
    @test Array(R3) == expected

    # In-place still rejects off-diagonal writes (Base-style ArgumentError).
    err_inp = @test_throws ArgumentError D .+= Float32(1)
    @test occursin("off-diagonal", sprint(showerror, err_inp.value))

    D_inp = Diagonal(NDArray(Float32[1, 2, 3]))
    err_mat_inp = @test_throws ArgumentError D_inp .+= A
    @test occursin("off-diagonal", sprint(showerror, err_mat_inp.value))

    # Structure-preserving scale still stays Diagonal without @allowscalar.
    D2 = Diagonal(NDArray(Float32[1, 2, 3]))
    D2 .*= Float32(4)
    @test Array(D2.diag) == Float32[4, 8, 12]
    D3 = D2 .* Float32(2)
    @test D3 isa Diagonal{Float32,<:NDArray{Float32,1}}
    @test Array(D3.diag) == Float32[8, 16, 24]

    # 1×1 has no off-diagonals: Base allows in-place densifying-classified ops.
    D1 = Diagonal(NDArray(Float32[5]))
    D1 .+= Float32(1)
    _host_diag_compare(Float32[6], D1.diag, Float32)
    D1 .*= Float32(2)
    _host_diag_compare(Float32[12], D1.diag, Float32)

    # 1×1 out-of-place densifies to NDArray (Base returns Matrix).
    R1 = Diagonal(NDArray(Float32[5])) .+ Float32(1)
    @test R1 isa NDArray{Float32,2}
    @test size(R1) == (1, 1)
    _host_matrix_compare(Float32[6;;], R1, Float32)
end

@testset "Diagonal * NDArray / mul! / lmul! / rmul!" begin
    @testset verbose=true for T in DIAGONAL_NUMERIC_TYPES
        n, m = 3, 4
        d = my_rand(T, n)
        dm = my_rand(T, m)
        Ah = my_rand(T, n, m)          # n×m
        As = my_rand(T, n, n)          # n×n
        vh = my_rand(T, n)
        Dh = Diagonal(d)
        Dm = Diagonal(dm)
        D = Diagonal(NDArray(d))
        D_m = Diagonal(NDArray(dm))
        A = NDArray(Ah)
        A_nd = NDArray(As)
        v = NDArray(vh)

        _host_diag_compare(Dh * vh, D * v, T)
        _host_matrix_compare(Dh * Ah, D * A, T)       # D (n)× A (n×m)
        _host_matrix_compare(Ah * Dm, A * D_m, T)     # A (n×m) × D (m)
        _host_matrix_compare(As * Dh, A_nd * D, T)    # square A*D

        C = cuNumeric.zeros(T, n, m)
        mul!(C, D, A)
        _host_matrix_compare(Dh * Ah, C, T)

        Cs = cuNumeric.zeros(T, n, n)
        mul!(Cs, A_nd, D)
        _host_matrix_compare(As * Dh, Cs, T)

        B = copy(A_nd)
        lmul!(D, B)
        _host_matrix_compare(Dh * As, B, T)

        B = copy(A_nd)
        rmul!(B, D)
        _host_matrix_compare(As * Dh, B, T)
    end
end

@testset "Diagonal \\ / / inv / ldiv! / rdiv!" begin
    # Floats: full path including singular CUDA-style behavior
    @testset verbose=true for T in DIAGONAL_FLOAT_TYPES
        n = 3
        d = _nonzero_diag(T, n)
        Ah = my_rand(T, n, n)
        vh = my_rand(T, n)
        Dh = Diagonal(d)
        D = Diagonal(NDArray(d))
        A = NDArray(Ah)
        v = NDArray(vh)

        _host_diag_compare(Dh \ vh, D \ v, T)
        _host_matrix_compare(Dh \ Ah, D \ A, T)
        _host_matrix_compare(Ah / Dh, A / D, T)
        allowscalar() do
            @test Matrix(inv(D)) ≈ Matrix(inv(Dh)) atol=atol(T) rtol=rtol(T)
        end

        B = copy(A)
        ldiv!(D, B)
        _host_matrix_compare(Dh \ Ah, B, T)

        B = copy(A)
        rdiv!(B, D)
        _host_matrix_compare(Ah / Dh, B, T)

        # Singular: zeros become Inf/NaN (no SingularException; that needs a host Bool)
        d0 = copy(d)
        d0[2] = zero(T)
        D0 = Diagonal(NDArray(d0))
        r = D0 \ v
        Di = inv(D0)
        Q = A / D0
        allowscalar() do
            @test any(isinf, Array(r)) || any(isnan, Array(r))
            @test any(isinf, Array(Di.diag)) || any(isnan, Array(Di.diag))
            @test any(isinf, Array(Q)) || any(isnan, Array(Q))
        end
    end

    # Complex: \ works via ./ ; inv/A/D blocked by missing __recip_type
    @testset verbose=true for T in DIAGONAL_COMPLEX_TYPES
        n = 3
        d = _nonzero_diag(T, n)
        Ah = my_rand(T, n, n)
        vh = my_rand(T, n)
        Dh = Diagonal(d)
        D = Diagonal(NDArray(d))
        A = NDArray(Ah)
        v = NDArray(vh)
        _host_diag_compare(Dh \ vh, D \ v, T)
        _host_matrix_compare(Dh \ Ah, D \ A, T)
    end

    # Integers with explicit allowpromotion for inv / \
    # `\` uses ./ → typically float(T) (Float64 for Int32/Int64).
    # `inv` / `A / D` use cuNumeric.__recip_type (Float32 for Int32, Float64 for Int64),
    # which is intentional NDArray promotion — not Base's float(Int32)==Float64.
    @testset verbose=true for T in (Int32, Int64)
        n = 3
        d = _nonzero_diag(T, n)
        Ah = my_rand(T, n, n)
        vh = my_rand(T, n)
        FT = float(T)
        RT = cuNumeric.__recip_type(T)
        Dh_div = Diagonal(FT.(d))
        Dh_inv = Diagonal(RT.(d))
        D = Diagonal(NDArray(d))
        A = NDArray(Ah)
        v = NDArray(vh)
        allowpromotion() do
            _host_diag_compare(Dh_div \ FT.(vh), D \ v, FT)
            _host_matrix_compare(Dh_div \ FT.(Ah), D \ A, FT)
            allowscalar() do
                @test Matrix(inv(D)) ≈ Matrix(inv(Dh_inv)) atol=atol(RT) rtol=rtol(RT)
            end
            _host_matrix_compare(RT.(Ah) / Dh_inv, A / D, RT)
        end
    end
end

@testset "Diagonal det" begin
    # Floats: det is prod of the diagonal (0D NDArray, not a Julia scalar)
    @testset verbose=true for T in DIAGONAL_FLOAT_TYPES
        d = my_rand(T, 3)
        Dh = Diagonal(d)
        D = Diagonal(NDArray(d))
        @test det(D) isa NDArray{<:Any,0}
        _host_scalar_compare(det(Dh), det(D), T)
        _host_scalar_compare(det(Matrix(D)), det(D), T)

        # 1×1
        d1 = T[T(5)]
        D1 = Diagonal(NDArray(d1))
        _host_scalar_compare(det(Diagonal(d1)), det(D1), T)
    end

    # Integers: prod may widen (e.g. Int32 → Int64); needs allowpromotion
    @testset verbose=true for T in (Int32, Int64)
        d = _nonzero_diag(T, 3)
        Dh = Diagonal(d)
        D = Diagonal(NDArray(d))
        allowpromotion() do
            _host_scalar_compare(det(Dh), det(D), T)
            _host_scalar_compare(det(Matrix(D)), det(D), T)
        end
        # 1×1
        D1 = Diagonal(NDArray(T[7]))
        allowpromotion() do
            _host_scalar_compare(T(7), det(D1), T)
        end
    end
end

@testset "NDArray ± Diagonal" begin
    @testset verbose=true for T in DIAGONAL_NUMERIC_TYPES
        n = 3
        d = my_rand(T, n)
        Ah = my_rand(T, n, n)
        Dh = Diagonal(d)
        D = Diagonal(NDArray(d))
        A = NDArray(Ah)
        _host_matrix_compare(Ah + Dh, A + D, T)
        _host_matrix_compare(Dh + Ah, D + A, T)
        _host_matrix_compare(Ah - Dh, A - D, T)
        _host_matrix_compare(Dh - Ah, D - A, T)
    end

    # Bool promotes under +
    @testset "Bool with allowpromotion" begin
        Ah = Bool[1 0; 0 1]
        d = Bool[true, true]
        A = NDArray(Ah)
        D = Diagonal(NDArray(d))
        allowpromotion() do
            allowscalar() do
                @test Array(A + D) == Ah + Diagonal(d)
                @test Array(A - D) == Ah - Diagonal(d)
            end
        end
    end
end

###### UniformScaling ######

# Prefer typed UniformScaling (one(T)*I) so narrow integers do not promote against
# Int64 λ from 2I / -I. Plain I (Bool λ) is fine for +/* on numeric arrays.

@testset "UniformScaling constructors / copyto!" begin
    @testset verbose=true for T in DIAGONAL_ARRAY_TYPES
        n = 3
        J1 = one(T) * I
        ref = Matrix{T}(J1, n, n)
        E = NDArray{T}(J1, n, n)
        _host_matrix_compare(ref, E, T)
        E2 = NDArray{T}(I, (n, n))  # Bool λ → ones on diagonal still
        _host_matrix_compare(Matrix{T}(I, n, n), E2, T)

        C = cuNumeric.zeros(T, n, n)
        copyto!(C, J1)
        _host_matrix_compare(ref, C, T)

        C = cuNumeric.ones(T, n, n)
        copyto!(C, zero(T) * I)
        _host_matrix_compare(zeros(T, n, n), C, T)

        # Rectangular scaled identity (skip Bool: Bool(2) is inexact)
        if T != Bool
            J2 = T(2) * I
            R = NDArray{T}(J2, 2, 3)
            allowscalar() do
                @test Array(R) == Matrix{T}(J2, 2, 3)
            end
            C = cuNumeric.zeros(T, 2, 3)
            copyto!(C, J2)
            allowscalar() do
                @test Array(C) == Matrix{T}(J2, 2, 3)
            end
        else
            R = NDArray{Bool}(I, 2, 3)
            allowscalar() do
                @test Array(R) == Matrix{Bool}(I, 2, 3)
            end
        end
    end

    # Untyped NDArray(I, ...) uses λ's type (Bool for I)
    E = NDArray(I, 2, 2)
    @test eltype(E) == Bool
    allowscalar() do
        @test Array(E) == Bool[1 0; 0 1]
    end
    E = NDArray(I, (2, 2))
    allowscalar() do
        @test Array(E) == Bool[1 0; 0 1]
    end
end

@testset "NDArray ± / * UniformScaling / one / oneunit" begin
    @testset verbose=true for T in DIAGONAL_NUMERIC_TYPES
        Ah = my_rand(T, 3, 3)
        A = NDArray(Ah)
        J1 = one(T) * I
        J2 = T(2) * I

        _host_matrix_compare(Ah + J1, A + J1, T)
        _host_matrix_compare(J1 + Ah, J1 + A, T)
        _host_matrix_compare(Ah * I, A * I, T)
        _host_matrix_compare(I * Ah, I * A, T)
        _host_matrix_compare(Ah * J2, A * J2, T)
        _host_matrix_compare(J2 * Ah, J2 * A, T)
        _host_matrix_compare(Matrix{T}(I, 3, 3), one(A), T)
        _host_matrix_compare(Matrix{T}(I, 3, 3), oneunit(A), T)

        # Subtraction / A+2I: signed & float/complex only (unsigned -one wraps; skip)
        if T <: Union{AbstractFloat,Complex} || (T <: Signed)
            _host_matrix_compare(Ah - J1, A - J1, T)
            _host_matrix_compare(J1 - Ah, J1 - A, T)
            _host_matrix_compare(Ah + J2, A + J2, T)
        end
    end

    # Plain I / 2I (Bool/Int64 λ) — natural API for floats & complex
    @testset verbose=true for T in (DIAGONAL_FLOAT_TYPES..., DIAGONAL_COMPLEX_TYPES...)
        Ah = my_rand(T, 3, 3)
        A = NDArray(Ah)
        _host_matrix_compare(Ah + I, A + I, T)
        _host_matrix_compare(I + Ah, I + A, T)
        _host_matrix_compare(Ah - I, A - I, T)
        _host_matrix_compare(I - Ah, I - A, T)
        _host_matrix_compare(Ah + 2I, A + 2I, T)
        _host_matrix_compare(Ah * (2I), A * (2I), T)
        _host_matrix_compare((2I) * Ah, (2I) * A, T)
    end

    # Bool: * I keeps Bool; ± I needs promotion
    @testset "Bool UniformScaling" begin
        Ah = Bool[1 0; 0 1]
        A = NDArray(Ah)
        allowscalar() do
            @test Array(A * I) == Ah
            @test eltype(A * I) == Bool
            @test Array(one(A)) == Ah
            @test Array(oneunit(A)) == Ah
        end
        allowpromotion() do
            allowscalar() do
                @test Array(A + I) == Ah + I
                @test Array(I - A) == I - Ah
            end
        end
    end
end

###### Diagonal ↔ UniformScaling ######

@testset "Diagonal ± / * UniformScaling / copyto!" begin
    @testset verbose=true for T in DIAGONAL_NUMERIC_TYPES
        d = my_rand(T, 3)
        Dh = Diagonal(d)
        D = Diagonal(NDArray(d))
        J1 = one(T) * I
        J2 = T(2) * I

        Di = D + J1
        @test Di isa Diagonal
        allowscalar() do
            @test Matrix(Di) ≈ Matrix(Dh + J1) atol=atol(eltype(Di)) rtol=rtol(eltype(Di))
            @test Matrix(J1 + D) ≈ Matrix(J1 + Dh) atol=atol(T) rtol=rtol(T)
            @test Matrix(D * I) ≈ Matrix(Dh * I) atol=atol(T) rtol=rtol(T)
            @test Matrix(I * D) ≈ Matrix(I * Dh) atol=atol(T) rtol=rtol(T)
            @test Matrix(D * J2) ≈ Matrix(Dh * J2) atol=atol(T) rtol=rtol(T)
        end

        if T <: Union{AbstractFloat,Complex} || (T <: Signed)
            allowscalar() do
                @test Matrix(D - J1) ≈ Matrix(Dh - J1) atol=atol(T) rtol=rtol(T)
                @test Matrix(J1 - D) ≈ Matrix(J1 - Dh) atol=atol(eltype((J1 - D).diag)) rtol=rtol(
                    eltype((J1 - D).diag)
                )
            end
        end

        copyto!(D, J1)
        allowscalar() do
            @test Array(D.diag) == ones(T, 3)
        end
        copyto!(D, zero(T) * I)
        allowscalar() do
            @test Array(D.diag) == zeros(T, 3)
        end
    end

    # Plain I on floats
    @testset verbose=true for T in DIAGONAL_FLOAT_TYPES
        d = my_rand(T, 3)
        Dh = Diagonal(d)
        D = Diagonal(NDArray(d))
        allowscalar() do
            @test Matrix(D + I) ≈ Matrix(Dh + I) atol=atol(T) rtol=rtol(T)
            @test Matrix(D - I) ≈ Matrix(Dh - I) atol=atol(T) rtol=rtol(T)
            @test Matrix(I - D) ≈ Matrix(I - Dh) atol=atol(T) rtol=rtol(T)
            @test Matrix(D * (2I)) ≈ Matrix(Dh * (2I)) atol=atol(T) rtol=rtol(T)
        end
    end

    @testset "Bool Diagonal + I with allowpromotion" begin
        D = Diagonal(NDArray(Bool[true, false]))
        allowpromotion() do
            Di = D + I
            @test Di isa Diagonal
            allowscalar() do
                @test Array(Di.diag) == [2, 1]
            end
        end
    end
end

###### Structured Diagonal vs densified Matrix path ######

# Compare Diagonal{NDArray} ops to the same math on densified host Matrices,
# and check that D + I stays Diagonal while A + I densifies to NDArray.

@testset "Diagonal vs dense" begin
    @testset verbose=true for T in DIAGONAL_FLOAT_TYPES
        n = 3
        d = _nonzero_diag(T, n)
        Ah = my_rand(T, n, n)
        vh = my_rand(T, n)
        Dh = Diagonal(d)
        Md = Matrix(Dh)                 # densified host counterpart
        D = Diagonal(NDArray(d))
        A = NDArray(Ah)
        v = NDArray(vh)

        _host_matrix_compare(Md * Ah, D * A, T)
        _host_matrix_compare(Ah * Md, A * D, T)
        _host_diag_compare(Md \ vh, D \ v, T)
        _host_matrix_compare(Ah / Md, A / D, T)
        allowscalar() do
            @test Matrix(inv(D)) ≈ inv(Md) atol=atol(T) rtol=rtol(T)
        end
        _host_matrix_compare(Ah + Md, A + D, T)

        Di = D + I
        @test Di isa Diagonal
        allowscalar() do
            @test Matrix(Di) ≈ Md + I atol=atol(T) rtol=rtol(T)
        end

        Ai = A + I
        @test Ai isa NDArray
        @test !(Ai isa Diagonal)
        _host_matrix_compare(Ah + I, Ai, T)
        _host_matrix_compare(Ah * I, A * I, T)
    end

    # Mul / add / structure for remaining numeric types (skip inv / div)
    @testset verbose=true for T in (DIAGONAL_COMPLEX_TYPES..., Int32, Int64)
        n = 3
        d = _nonzero_diag(T, n)
        Ah = my_rand(T, n, n)
        Dh = Diagonal(d)
        Md = Matrix(Dh)
        D = Diagonal(NDArray(d))
        A = NDArray(Ah)
        J1 = one(T) * I

        _host_matrix_compare(Md * Ah, D * A, T)
        _host_matrix_compare(Ah * Md, A * D, T)
        _host_matrix_compare(Ah + Md, A + D, T)

        Di = D + J1
        @test Di isa Diagonal
        allowscalar() do
            @test Matrix(Di) ≈ Md + Matrix{T}(J1, n, n) atol=atol(T) rtol=rtol(T)
        end

        Ai = A + J1
        @test Ai isa NDArray
        @test !(Ai isa Diagonal)
        _host_matrix_compare(Ah + J1, Ai, T)
    end
end

###### Native LinearAlgebra API (supported Diagonal paths) ######

@testset "Diagonal eigen / eigvals native" begin
    @testset verbose=true for T in DIAGONAL_FLOAT_TYPES
        d = T[T(3), T(1), T(2)]
        Dh = Diagonal(d)
        D = Diagonal(NDArray(d))

        # eigvals is a copy of the diagonal (NDArray)
        λ = eigvals(D)
        @test λ isa NDArray{T,1}
        _host_diag_compare(eigvals(Dh), λ, T)
        @test λ !== D.diag

        # unsorted eigen: values == diag copy, vectors == I (NDArray)
        F = eigen(D)
        @test F.values isa NDArray{T,1}
        _host_diag_compare(d, F.values, T)
        @test F.vectors isa NDArray{T,2}
        _host_matrix_compare(Matrix{T}(I, 3, 3), F.vectors, T)

        @test eigvecs(D) isa NDArray{T,2}
        _host_matrix_compare(Matrix{T}(I, 3, 3), eigvecs(D), T)
    end
end

@testset "Diagonal native reductions / predicates / norms" begin
    @testset verbose=true for T in DIAGONAL_FLOAT_TYPES
        d = abs.(my_rand(T, 3)) .+ one(T)   # positive → isposdef
        Dh = Diagonal(d)
        D = Diagonal(NDArray(d))

        _host_scalar_compare(tr(Dh), tr(D), T)
        _host_scalar_compare(sum(Dh), sum(D), T)
        _host_scalar_compare(zero(T), prod(D), T)  # n>1 off-diagonals
        _host_scalar_compare(maximum(Dh), maximum(D), T)
        _host_scalar_compare(minimum(Dh), minimum(D), T)
        _host_bool_compare(isposdef(Dh), isposdef(D))
        _host_bool_compare(true, issymmetric(D))
        _host_bool_compare(true, ishermitian(D))
        @test isdiag(D)
        _host_bool_compare(false, iszero(D))
        _host_bool_compare(false, isone(D))
        _host_bool_compare(true, iszero(Diagonal(cuNumeric.zeros(T, 3))))
        _host_bool_compare(true, isone(Diagonal(cuNumeric.ones(T, 3))))
        _host_bool_compare(true, istriu(D))
        _host_bool_compare(true, istril(D))
        _host_bool_compare(false, istriu(D, 1))
        _host_bool_compare(false, istril(D, -1))

        _host_scalar_compare(opnorm(Dh), opnorm(D), T)
        _host_scalar_compare(norm(Dh), norm(D), T)
        _host_scalar_compare(cond(Dh), cond(D), T)
        _host_scalar_compare(logdet(Dh), logdet(D), T)

        # matrix functions via f.(diag) broadcast (Base Diagonal methods)
        allowscalar() do
            @test Matrix(sqrt(D)) ≈ Matrix(sqrt(Dh)) atol=atol(T) rtol=rtol(T)
            @test Matrix(exp(D)) ≈ Matrix(exp(Dh)) atol=atol(T) rtol=rtol(T)
        end
    end
end
