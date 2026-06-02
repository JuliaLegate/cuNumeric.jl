using Random: Random
using cuNumeric: cuNumeric

include("benchmarks.jl")
include("parse_benchmarks.jl")

function benchmark(b::AbstractBenchmark, gs::GlobalSettings, arrs)
    GC.gc(; full=true)

    for idx in range(1, gs.n_iter + gs.n_warmup)
        if idx == gs.n_warmup + 1
            start_time = get_time_microseconds()
        end

        run!(b, arrays...)
    end
    total_time_μs = get_time_microseconds() - start_time
    mean_time_ms = total_time_μs / (gs.n_iter * 1e3)
    gflops = total_flops(N, M) / (mean_time_ms * 1e6)

    GC.gc(; full=true)

    return mean_time_ms, gflops
end

function run_all_benchmarks()
    global_settings, benchmarks = parse_config("benchmarks.toml")

    @show global_settings
    @show benchmarks

    cunumeric_results = BenchmarkResult[]
    cuda_results = BenchmarkResult[]

    for b in benchmarks
        println("================================")
        println(data(b))
        println("================================")

        cn_times_ms = Vector{Float64}(undef, global_settings.n_trial)
        cn_gflops = Vector{Union{Missing,Float64}}(undef, global_settings.n_trial)

        cuda_times_ms = Vector{Float64}(undef, global_settings.n_trial)
        cuda_gflops = Vector{Union{Missing,Float64}}(undef, global_settings.n_trial)

        for i in 1:global_settings.n_trial
            arrs_julia = initialize_cpu(b)

            arrs_cunumeric = # TODO
                cn_times_ms[i], cn_gflops[i] = benchmark(b, arrs_cunumeric...)
            push

            if gs.n_gpu == 1
                arrs_cuda = # TODO
                    cuda_times_ms[i], cuda_gflops[i] = benchmark(b, arrs_cuda...)
                push!(cuda_results, res_cuda)
            end
        end

        cn_result = BenchmarkResult(cn_times_ms, cn_gflops, b)
        cuda_result = BenchmarkResult(cuda_times_ms, cuda_gflops, b)

        push!(cunumeric_results, cn_result)
        push!(cuda_results, cuda_result)
    end

    # Call the `save` function for the cuda_results
    # This function is not implemeneted as I was not sure how to do it

end
