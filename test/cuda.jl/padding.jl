#= Copyright 2025 Northwestern University,
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

#= Purpose of test: cuda
    -- Validate custom-kernel padding, synchronization, and lifetime management
=#

using CUDACore: blockDim, blockIdx, threadIdx
import CUDACore: i32

cuNumeric.Experimental(true)

function padding_add(a, b, c, N)
    i = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    if i <= N
        @inbounds c[i] = a[i] + b[i]
    end
    return nothing
end

function padding_mul(a, c, b, N)
    i = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    if i <= N
        @inbounds b[i] = a[i] * c[i]
    end
    return nothing
end

function cuda_padding_lifetime()
    N = 1_000_000
    M = N - 2
    threads = 256
    blocks = cld(M, threads)
    initial_bytes = cuNumeric.current_device_bytes[]

    a = cuNumeric.ones(Float32, N)
    b = cuNumeric.ones(Float32, N)
    c = cuNumeric.zeros(Float32, M)
    task = cuNumeric.@cuda_task padding_add(a, b, c, UInt32(M))
    unpadded_bytes = cuNumeric.current_device_bytes[]

    try
        @test @inferred(
            cuNumeric.launch(
                task, (a, b), c, UInt32(M); threads=threads, blocks=blocks
            )
        ) === nothing
        padded_bytes = cuNumeric.current_device_bytes[]

        @test @inferred(cuNumeric._launch_shape(c)) == (N,)
        @test @inferred(cuNumeric._sync_to_launch_padding!(c)) === nothing
        @test @inferred(cuNumeric._sync_from_launch_padding!(c)) === nothing

        accounting_ok = true
        for _ in 1:15
            cuNumeric.@launch task=task threads=threads blocks=blocks inputs=(a, b) outputs=c scalars=UInt32(
                M
            )
            accounting_ok &= cuNumeric.current_device_bytes[] == padded_bytes
        end

        @test padded_bytes > unpadded_bytes
        @test accounting_ok
        @test size(c) == (M,)
        @test all(Array(c) .== 2.0f0)
    finally
        cuNumeric.destroy!(a)
        cuNumeric.destroy!(b)
        cuNumeric.destroy!(c)
    end
    @test cuNumeric.current_device_bytes[] == initial_bytes
end

function cuda_padding_slice_output()
    N = 4096
    M = N - 2
    threads = 256
    blocks = cld(M, threads)
    initial_bytes = cuNumeric.current_device_bytes[]

    a = cuNumeric.ones(Float32, N)
    b = cuNumeric.ones(Float32, N)
    parent = cuNumeric.zeros(Float32, N)
    output = parent[1:M]
    task = cuNumeric.@cuda_task padding_add(a, b, output, UInt32(M))
    unpadded_bytes = cuNumeric.current_device_bytes[]

    try
        @test @inferred(
            cuNumeric.launch(
                task, (a, b), output, UInt32(M); threads=threads, blocks=blocks
            )
        ) === nothing
        padded_bytes = cuNumeric.current_device_bytes[]
        @test @inferred(cuNumeric._launch_shape(output)) == (N,)
        @test @inferred(cuNumeric._sync_from_launch_padding!(output)) === nothing

        task = cuNumeric.@cuda_task padding_mul(a, output, b, UInt32(M))
        @test @inferred(
            cuNumeric.launch(
                task, (a, output), b, UInt32(M); threads=threads, blocks=blocks
            )
        ) === nothing
        values = Array(parent)
        product = Array(b)

        @test padded_bytes > unpadded_bytes
        @test cuNumeric.current_device_bytes[] == padded_bytes
        @test all(values[1:M] .== 2.0f0)
        @test all(product[1:M] .== 2.0f0)
        @test values[end] == 0.0f0
    finally
        cuNumeric.destroy!(output)
        cuNumeric.destroy!(parent)
        cuNumeric.destroy!(a)
        cuNumeric.destroy!(b)
    end
    @test cuNumeric.current_device_bytes[] == initial_bytes
end

Base.@noinline function drop_padded_arrays()
    N = 4096
    M = N - 2
    a = cuNumeric.ones(Float32, N)
    b = cuNumeric.ones(Float32, N)
    c = cuNumeric.zeros(Float32, M)
    task = cuNumeric.@cuda_task padding_add(a, b, c, UInt32(M))
    cuNumeric.@launch task=task threads=256 blocks=cld(M, 256) inputs=(a, b) outputs=c scalars=UInt32(
        M
    )
    return nothing
end

function cuda_padding_finalizer()
    GC.gc(true)
    cuNumeric.drain_pending_frees!()
    baseline = cuNumeric.current_device_bytes[]

    drop_padded_arrays()
    allocated = cuNumeric.current_device_bytes[]
    GC.gc(true)
    GC.gc(true)
    cuNumeric.drain_pending_frees!()

    @test allocated > baseline
    @test cuNumeric.current_device_bytes[] == baseline
end

try
    @testset "Custom CUDA padding" begin
        cuda_padding_lifetime()
        cuda_padding_slice_output()
        cuda_padding_finalizer()
    end
finally
    cuNumeric.Experimental(false)
end
