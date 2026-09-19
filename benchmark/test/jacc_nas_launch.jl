# Reproduce CUDA's mutable LaunchSpec behavior without requiring a GPU/JACC.
module JACCNASLaunchTest
using Test
const ModelWorkerConfig = Main.ModelWorkerConfig

module JACC
    Base.@kwdef mutable struct LaunchSpec
        sync::Bool = true
        shmem_size::Int = -1
        threads::Int = 0
        blocks::Int = 0
    end
    launch_spec(; kwargs...) = LaunchSpec(; kwargs...)
    function parallel_for(spec, n, kernel, args...)
        @assert !spec.sync && spec.shmem_size == 0
        spec.threads == 0 && (spec.threads = min(n, 32))
        spec.blocks == 0 && (spec.blocks = cld(n, spec.threads))
        for i in 1:min(n, spec.threads * spec.blocks)
            kernel(i, args...)
        end
    end
end

include("../src/jacc/benchmarks/nas/mg.jl")

@testset "JACC MG launch coverage changes with grid size" begin
    # A retained spec from the first (coarse) grid would leave most fine cells
    # untouched. Repeat in both directions, including non-block-aligned sizes.
    for n in (4, 10, 34, 18, 4, 34)
        out = fill(NaN, n, n, n)
        jacc_mg_fill!(nothing, out)
        @test all(iszero, out)
    end
end
end
