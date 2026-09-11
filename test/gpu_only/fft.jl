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

using FFTW

const _FFT_TEST_TYPES = Base.uniontypes(cuNumeric._FFT_ACCEPTED)
const _FFT_INPLACE_TYPES = Base.uniontypes(cuNumeric.SUPPORTED_COMPLEX_TYPES)

_fft_out_eltype(::Type{T}) where {T} = cuNumeric._fft_eltype(T)

function _reference_fft(x::AbstractArray, dims=ntuple(identity, ndims(x)))
    return FFTW.fft(_fft_out_eltype(eltype(x)).(x), dims)
end

function _reference_ifft(x::AbstractArray, dims=ntuple(identity, ndims(x)))
    return FFTW.ifft(_fft_out_eltype(eltype(x)).(x), dims)
end

function _fft_compare(got::NDArray, expected; rtol, atol)
    return @allowscalar isapprox(Array(got), expected; rtol=rtol, atol=atol)
end

@testset verbose = true "fft/ifft" begin
    @testset verbose = true for T in _FFT_TEST_TYPES
        OT = _fft_out_eltype(T)
        rtol_t = rtol(OT)
        atol_t = atol(OT)

        @testset "1d" begin
            x_cpu = my_rand(T, 16)
            x = cuNumeric.NDArray(x_cpu)
            y = fft(x)
            @test eltype(y) === OT
            @test _fft_compare(y, _reference_fft(x_cpu); rtol=rtol_t, atol=atol_t)
            @test _fft_compare(
                ifft(y), _reference_ifft(_reference_fft(x_cpu)); rtol=rtol_t, atol=atol_t
            )
        end

        @testset "2d all dims" begin
            x_cpu = my_rand(T, 8, 12)
            x = cuNumeric.NDArray(x_cpu)
            y = fft(x)
            @test eltype(y) === OT
            @test _fft_compare(y, _reference_fft(x_cpu); rtol=rtol_t, atol=atol_t)
            @test _fft_compare(
                ifft(y), _reference_ifft(_reference_fft(x_cpu)); rtol=rtol_t, atol=atol_t
            )
        end

        @testset "2d dims=1" begin
            x_cpu = my_rand(T, 8, 12)
            x = cuNumeric.NDArray(x_cpu)
            y = fft(x, 1)
            @test _fft_compare(y, _reference_fft(x_cpu, (1,)); rtol=rtol_t, atol=atol_t)
            @test _fft_compare(
                ifft(y, 1), _reference_ifft(_reference_fft(x_cpu, (1,)), (1,));
                rtol=rtol_t, atol=atol_t,
            )
        end

        @testset "2d dims=2" begin
            x_cpu = my_rand(T, 8, 12)
            x = cuNumeric.NDArray(x_cpu)
            y = fft(x, 2)
            @test _fft_compare(y, _reference_fft(x_cpu, (2,)); rtol=rtol_t, atol=atol_t)
            @test _fft_compare(
                ifft(y, 2), _reference_ifft(_reference_fft(x_cpu, (2,)), (2,));
                rtol=rtol_t, atol=atol_t,
            )
        end
    end
end

@testset verbose = true "fft!/ifft!" begin
    @testset verbose = true for T in _FFT_INPLACE_TYPES
        rtol_t = rtol(T)
        atol_t = atol(T)
        x_cpu = my_rand(T, 16)
        expected = _reference_fft(x_cpu)

        x = cuNumeric.NDArray(copy(x_cpu))
        y = fft!(x)
        @test y === x
        @test _fft_compare(x, expected; rtol=rtol_t, atol=atol_t)

        z = ifft!(x)
        @test z === x
        @test _fft_compare(x, x_cpu; rtol=rtol_t, atol=atol_t)
    end

    @testset "fft! rejects non-complex" begin
        for T in (Float32, Float64, Int32, Bool)
            x = cuNumeric.NDArray(my_rand(T, 8))
            @test_throws ArgumentError fft!(x)
            @test_throws ArgumentError ifft!(x)
        end
    end
end

@testset verbose = true "batched_fft" begin
    @testset verbose = true for T in _FFT_TEST_TYPES
        OT = _fft_out_eltype(T)
        rtol_t = rtol(OT)
        atol_t = atol(OT)

        @testset "1d signals (b, n)" begin
            x_cpu = my_rand(T, 4, 16)
            x = cuNumeric.NDArray(x_cpu)
            y = batched_fft(x)
            @test eltype(y) === OT
            @test _fft_compare(y, _reference_fft(x_cpu, (2,)); rtol=rtol_t, atol=atol_t)
            @test _fft_compare(
                batched_ifft(y), _reference_ifft(_reference_fft(x_cpu, (2,)), (2,));
                rtol=rtol_t, atol=atol_t,
            )
        end

        @testset "2d fields (b, n, m)" begin
            x_cpu = my_rand(T, 3, 8, 12)
            x = cuNumeric.NDArray(x_cpu)
            y = batched_fft(x)
            @test _fft_compare(y, _reference_fft(x_cpu, (2, 3)); rtol=rtol_t, atol=atol_t)
            @test _fft_compare(
                batched_ifft(y), _reference_ifft(_reference_fft(x_cpu, (2, 3)), (2, 3));
                rtol=rtol_t, atol=atol_t,
            )
        end
    end

    @testset verbose = true for T in _FFT_INPLACE_TYPES
        rtol_t = rtol(T)
        atol_t = atol(T)
        x_cpu = my_rand(T, 4, 16)
        expected = _reference_fft(x_cpu, (2,))
        x = cuNumeric.NDArray(copy(x_cpu))
        y = batched_fft!(x)
        @test y === x
        @test _fft_compare(x, expected; rtol=rtol_t, atol=atol_t)
        z = batched_ifft!(x)
        @test z === x
        @test _fft_compare(x, x_cpu; rtol=rtol_t, atol=atol_t)
    end

    @testset "rejects 1d" begin
        x = cuNumeric.NDArray(my_rand(ComplexF32, 8))
        @test_throws ArgumentError batched_fft(x)
        @test_throws ArgumentError batched_fft!(x)
    end
end
