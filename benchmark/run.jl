import Random
import cuNumeric

include("benchmarks.jl")

function work(b::AbstractBenchmark, N_GPU, arrs_cpu...)

    run!(b, arrs_cunumeric...)

    GC.gc(full = true)

    if N_GPU == 1
        arrs_cuda = ...
        run!(b, arrs_cunumeric...)
    end

    # Reset state in between
    GC.gc(full = true)
end

function run_all_benchmarks()

    global_settings, benchmarks = parse_config("benchmarks.toml")

    @show global_settings
    @show benchmarks

    for b in benchmarks
        println("================================")
        println(data(b))
        println("================================")

        arrs = init(benchmark)

        #TODO FIX

        arrs_cunumeric =
        run!(b, arrs_cunumeric...)

        # Reset state in between
        GC.gc(full = true)

        if N_GPU == 1
            arrs_cuda = ...
            run!(b, arrs_cunumeric...)
        end

        # Reset state in between
        GC.gc(full = true)
    end

end


function run_sgemm_benchmark(N)
    include("sgemm.jl")
    name = "SGEMM"
end

function run_monte_carlo_benchmark(N)
    include("monte_carlo.jl")
    name = "Monte_Carlo_Integration"
end
