#= Copyright 2025 Northwestern University,
 *                   Carnegie Mellon University University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSEend-2.0
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
    -- Register various custom kernels using CUDA.jl
=#

using CUDACore: blockDim, blockIdx, threadIdx
import CUDACore: i32

cuNumeric.Experimental(true)

function kernel_add(a, b, c, N)
    i = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    if i <= N
        @inbounds c[i] = a[i] + b[i]
    end
    return nothing
end

# Test a second kernel with `c` as an input and `b` as the output.
function kernel_mul(a, c, b, N)
    i = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    if i <= N
        @inbounds b[i] = a[i] * c[i]
    end
    return nothing
end

function cuda_binaryop(max_diff)
    N = 1024
    threads = 256
    blocks = cld(N, threads)
    FT = Float32

    a = cuNumeric.zeros(FT, N)
    b = cuNumeric.zeros(FT, N)
    c = cuNumeric.zeros(FT, N)

    a_cpu = rand(FT, N)
    b_cpu = rand(FT, N)
    c_cpu = zeros(FT, N)

    allowscalar() do
        for i in 1:N
            a[i] = a_cpu[i]
            b[i] = b_cpu[i]
        end
    end

    # get results on CPU for comparison
    for i in 1:N
        c_cpu[i] = a_cpu[i] + b_cpu[i]
    end

    task = cuNumeric.@cuda_task kernel_add(a, b, c, UInt32(1))
    cuNumeric.@launch task=task threads=threads blocks=blocks inputs=(a, b) outputs=c scalars=UInt32(
        N
    )

    @test @allowscalar cuNumeric.compare(c, c_cpu, max_diff, max_diff)

    b_cpu .= a_cpu .* c_cpu

    task = cuNumeric.@cuda_task kernel_mul(a, c, b, UInt32(1))
    cuNumeric.@launch task=task threads=threads blocks=blocks inputs=(a, c) outputs=b scalars=UInt32(
        N
    )

    @test @allowscalar cuNumeric.compare(b, b_cpu, max_diff, max_diff)
end

function kernel_sin(a, b, N)
    i = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    if i <= N
        @inbounds b[i] = @fastmath sin(a[i])
    end
    return nothing
end

function cuda_unaryop(max_diff)
    N = 1024
    threads = 64
    blocks = cld(N, threads)
    FT = Float32

    a = cuNumeric.zeros(FT, N)
    b = cuNumeric.zeros(FT, N)

    a_cpu = rand(FT, N)
    b_cpu = zeros(FT, N)

    allowscalar() do
        for i in 1:N
            a[i] = a_cpu[i]
        end
    end

    # get results on CPU for comparison
    for i in 1:N
        b_cpu[i] = sin(a_cpu[i])
    end

    task = cuNumeric.@cuda_task kernel_sin(a, b, UInt32(N))
    # TODO explore getting inplace ops working.
    cuNumeric.@launch task=task threads=threads blocks=blocks inputs=a outputs=b scalars=UInt32(N)

    @test @allowscalar cuNumeric.compare(b, b_cpu, max_diff, max_diff)
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
    task = cuNumeric.@cuda_task kernel_add(a, b, c, UInt32(M))
    unpadded_bytes = cuNumeric.current_device_bytes[]

    try
        cuNumeric.@launch task=task threads=threads blocks=blocks inputs=(a, b) outputs=c scalars=UInt32(
            M
        )
        padded_bytes = cuNumeric.current_device_bytes[]

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
    task = cuNumeric.@cuda_task kernel_add(a, b, output, UInt32(M))
    unpadded_bytes = cuNumeric.current_device_bytes[]

    try
        cuNumeric.@launch task=task threads=threads blocks=blocks inputs=(a, b) outputs=output scalars=UInt32(
            M
        )
        padded_bytes = cuNumeric.current_device_bytes[]
        task = cuNumeric.@cuda_task kernel_mul(a, output, b, UInt32(M))
        cuNumeric.@launch task=task threads=threads blocks=blocks inputs=(a, output) outputs=b scalars=UInt32(
            M
        )
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
    task = cuNumeric.@cuda_task kernel_add(a, b, c, UInt32(M))
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
    @testset "Custom CUDA kernels" begin
        cuda_binaryop(1.0f-5)
        cuda_unaryop(1.0f-5)
        cuda_padding_lifetime()
        cuda_padding_slice_output()
        cuda_padding_finalizer()
    end
finally
    cuNumeric.Experimental(false)
end
