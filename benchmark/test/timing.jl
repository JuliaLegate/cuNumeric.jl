struct TimingProbe <: AbstractBenchmark{Float32}
    events::Vector{Symbol}
end
struct GrayScottTimingProbe <: AbstractGrayScott{Float32}
    events::Vector{Symbol}
end
const TimingProbes = Union{TimingProbe,GrayScottTimingProbe}
initialize(b::TimingProbes; mod=Base) = (push!(b.events, :initialize); ())
run!(b::TimingProbes) = push!(b.events, :run)
total_flops(::TimingProbes) = 6000
name(::TimingProbes) = "timing probe"

@testset "Iteration completion policy" begin
    for B in values(BENCHMARKS)
        b = build_benchmark(B, Float32, 32, 32)
        @test fence_each_iteration(b) == !(b isa AbstractGrayScott)
    end
    for B in (TimingProbe, GrayScottTimingProbe), warmup in (0, 2)
        events = Symbol[]
        b = B(events)
        ticks = Ref(0)
        clock() = (push!(events, :clock); ticks[] += 6000)
        synchronize() = push!(events, :sync)
        gs = GlobalSettings(; n_warmup=warmup, n_iter=3)
        step = fence_each_iteration(b) ? [:run, :sync] : [:run]
        expected = vcat([:initialize], repeat(step, warmup), [:clock],
                        repeat(step, 3), [:clock])
        # Also exercises forwarding the backend callback through run_benchmark.
        result = run_benchmark(b, gs; mod=Base, clock, synchronize)
        @test events == expected
        @test result.times_ms == [2.0]
        @test result.gflops == [0.003]
    end
end
