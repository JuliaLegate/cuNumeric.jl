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

@testset "Zero-Copy Verification" begin
    A = rand(Float64, 4, 4)
    NA = NDArray(A)

    @allowscalar begin
        @test all(A .== Array(NA))
    end

    @test pointer(A) == cuNumeric.get_ptr(NA)

    # modify julia array, verify ndarray sees it
    A[1, 1] = 99.0
    @allowscalar begin
        @test NA[1, 1] == 99.0
    end

    # modify ndarray, verify julia array sees it
    @allowscalar begin
        NA[2, 2] = 88.0
    end
    @test A[2, 2] == 88.0
end

@testset "Lifetime Protection" begin
    function create_attached_ndarray()
        local_A = rand(Float32, 100)
        local_A[1] = 1.23f0
        return NDArray(local_A), local_A[1]
    end

    NA, expected_val = create_attached_ndarray()

    # force gc to try and collect the local array
    GC.gc(true)
    GC.gc(true) # do it again
    GC.gc(true) # and again lol

    # data should still be intact via parent reference
    @allowscalar begin
        @test NA[1] == expected_val
    end

    # operations should still work
    NA2 = NA .* 2.0f0
    @allowscalar begin
        @test NA2[1] == expected_val * 2.0f0
    end
end

@testset "Type Conversion Lifetime" begin
    function create_typed_ndarray()
        I = collect(1:10)
        return NDArray{Float64}(I)
    end

    NA = create_typed_ndarray()

    # the temporary Float64 array should be kept alive by NA.parent
    GC.gc(true)
    GC.gc(true) 
    GC.gc(true) 

    @allowscalar begin
        @test NA[1] == 1.0
        @test NA[10] == 10.0
    end

    # modification should work on the attached temporary
    @allowscalar begin
        NA[1] = 42.0
        @test NA[1] == 42.0
    end
end
