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
        @allowscalar @test cuNumeric.compare(expected, output, 0, 0)

        if length(shape) > 1
            col_output = cuNumeric.h5read(path, dataset; layout=:col)
            col_expected = permutedims(expected, reverse(1:length(shape)))
            @test size(col_output) == reverse(shape)
            @allowscalar @test cuNumeric.compare(col_expected, col_output, 0, 0)
        end
    end
end

@testset "HDF5" begin
    for T in (Float32, Float64, Int32, Int64)
        @testset "$T $shape" for shape in ((7,), (3, 4), (2, 3, 4))
            test_hdf5_roundtrip(T, shape)
        end
    end
end
