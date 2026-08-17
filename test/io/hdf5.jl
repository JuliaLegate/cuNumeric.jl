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
 *            Nader Rahal <naderrahhal2026@u.northwestern.edu>
=#

function test_hdf5_roundtrip(::Type{T}, shape::Tuple) where {T}
    expected = reshape(T.(1:prod(shape)), shape)
    input = cuNumeric.zeros(T, shape...)
    @allowscalar for index in CartesianIndices(shape)
        input[Tuple(index)...] = expected[index]
    end

    return mktempdir() do dir
        path = joinpath(dir, "roundtrip.h5")
        dataset = "values"

        cuNumeric.h5write(path, dataset, input)
        cuNumeric.Legate.runtime_sync()
        @test isfile(path)

        output = cuNumeric.h5read(path, dataset; layout=:row)
        @test eltype(output) == T
        @test size(output) == shape
        @allowscalar @test safe_compare(expected, output, 0, 0)

        if length(shape) > 1
            col_output = cuNumeric.h5read(path, dataset; layout=:col)
            col_expected = permutedims(expected, reverse(1:length(shape)))
            @test size(col_output) == reverse(shape)
            @allowscalar @test safe_compare(col_expected, col_output, 0, 0)
        end
    end
end

@testset "HDF5" begin
    for T in (Float32, Float64, Int32, Int64)
        @testset "$T $shape" for shape in ((7,), (3, 4), (2, 3, 4))
            test_hdf5_roundtrip(T, shape)
        end
    end

    @testset "host-side path checks" begin
        mktempdir() do dir
            arr = cuNumeric.ones(Float32, 4, 3)
            path = joinpath(dir, "values.h5")

            @test_throws ArgumentError cuNumeric.h5read(joinpath(dir, "missing.h5"), "values")
            @test_throws ArgumentError cuNumeric.h5read(dir, "values")
            @test_throws ArgumentError cuNumeric.h5write(dir, "values", arr)
            @test_throws ArgumentError cuNumeric.h5write(path, "", arr)
            @test_throws ArgumentError cuNumeric.h5read(path, "")

            stub = joinpath(dir, "stub.h5")
            write(stub, "not hdf5")
            @test_throws ArgumentError cuNumeric.h5read(stub, "values")

            # A leftover empty file is what aborted HDF5CombineVDS.
            leftover = joinpath(dir, "leftover.h5")
            write(leftover, UInt8[])
            cuNumeric.h5write(leftover, "values", arr)
            cuNumeric.Legate.runtime_sync()
            @test cuNumeric._is_hdf5_file(leftover)
            @test size(cuNumeric.h5read(leftover, "values")) == size(arr)
            cuNumeric.Legate.runtime_sync()

            # A second write to the same path must replace, not abort.
            arr2 = cuNumeric.zeros(Float32, 4, 3)
            cuNumeric.h5write(leftover, "values", arr2)
            cuNumeric.Legate.runtime_sync()
            @allowscalar @test safe_compare(
                zeros(Float32, 4, 3), cuNumeric.h5read(leftover, "values"), 0, 0
            )
        end
    end
end
