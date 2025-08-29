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
using cuNumeric
using CUDA
import CUDA: i32
using Test


function kernel_add(a, b, c, N)
    i = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    if i <= N
        @inbounds c[i] = a[i] + b[i]
    end
    return nothing
end

# testing a second kernel
# on purpose switching inputs and outputs
function kernel_mul(a, b, c, N)
    i = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    if i <= N
        @inbounds b[i] = a[i] * c[i]
    end
    return nothing
end

function binaryop(max_diff)
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

    for i in 1:N
        a[i] = a_cpu[i]
        b[i] = b_cpu[i]
    end

    # get results on CPU for comparison
    for i in 1:N
        c_cpu[i] = a_cpu[i] + b_cpu[i]
    end

    task = cuNumeric.@cuda_task kernel_add(a, b, c, UInt32(1))
    cuNumeric.@launch task=task threads=threads blocks=blocks inputs=(a, b) outputs=c scalars=UInt32(
        N
    )

    @test cuNumeric.compare(c, c_cpu, max_diff)

    for i in 1:N
        b[i] = a[i] * c[i]
    end

    task = cuNumeric.@cuda_task kernel_mul(a, b, c, UInt32(1))
    cuNumeric.@launch task=task threads=threads blocks=blocks inputs=(a, c) outputs=b scalars=UInt32(
        N
    )

    @test cuNumeric.compare(b, b_cpu, max_diff)
end

function kernel_sin(a, b, N)
    i = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    if i <= N
        @inbounds b[i] = @fastmath sin(a[i])
    end
    return nothing
end

function unaryop(max_diff)
    N = 1024
    threads = 64
    blocks = cld(N, threads)
    FT = Float32

    a = cuNumeric.zeros(FT, N)
    b = cuNumeric.zeros(FT, N)

    a_cpu = rand(FT, N)
    b_cpu = zeros(FT, N)

    for i in 1:N
        a[i] = a_cpu[i]
    end

    # get results on CPU for comparison
    for i in 1:N
        b_cpu[i] = sin(a_cpu[i])
    end

    task = cuNumeric.@cuda_task kernel_sin(a, b, UInt32(1))
    # TODO explore getting inplace ops working. 
    cuNumeric.@launch task=task threads=threads blocks=blocks inputs=a outputs=b scalars=UInt32(N)

    @test cuNumeric.compare(b, b_cpu, max_diff)
end

unaryop(Float32(1e-4))
binaryop(Float32(1e-4))
