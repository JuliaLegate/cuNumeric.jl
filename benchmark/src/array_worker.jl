# Shared protocol for the cuNumeric and CUDA.jl array workers. Before including
# this file, the model entrypoint imports its packages, loads `core.jl` and the
# shared benchmarks, loads any model-specific benchmark methods, and defines
# `array_backend_entry`.

length(ARGS) in (11, 12) || error(
    "array worker args: <gpus> <name> <T> <N> <M> <n_iter> <n_warmup> " *
    "<n_trial> <check_correctness> <n_correctness_iter> <flops> [kwargs TOML]",
)

using Printf, TOML
using Statistics

parse_worker_type(s) = get(Dict("Float32"=>Float32, "Float64"=>Float64), s) do
    return error("Unsupported element type '$s'; known: Float32, Float64")
end

function run_array_worker(args=ARGS)
    gpus = parse(Int, args[1])
    name = args[2]
    T_name = args[3]
    N, M = parse(Int, args[4]), parse(Int, args[5])
    n_iter, n_warmup, n_trial = parse(Int, args[6]), parse(Int, args[7]), parse(Int, args[8])
    check_correctness = parse(Bool, args[9])
    n_correctness_iter = parse(Int, args[10])
    parse(Float64, args[11]) # validate the uniform protocol's FLOP field
    gpus > 0 || error("gpus must be positive")
    n_iter > 0 && n_warmup >= 0 && n_trial > 0 ||
        error("invalid trial/iteration count")
    haskey(BENCHMARKS, name) || error(
        "No benchmark registered for '$name'. Known: " *
        join(sort!(collect(keys(BENCHMARKS))), ", "),
    )

    backend = array_backend_entry()
    T = parse_worker_type(T_name)
    kwargs = length(args)==12 ? Dict(Symbol(k)=>v for (k, v) in TOML.parse(args[12])) : Dict()
    benchmark = build_benchmark(BENCHMARKS[name], T, N, M; kwargs...)
    fused = backend.fused()
    default_save_as = fused ? backend.save_as : "$(backend.save_as)_nofusion"
    default_label = fused ? backend.label : "$(backend.label) (no fusion)"
    save_as = benchmark_backend_save_as(
        benchmark, string(backend.id), default_save_as
    )
    label = benchmark_backend_label(benchmark, string(backend.id), default_label)
    settings = GlobalSettings(;
        n_warmup, n_iter, n_trial, n_gpu=gpus, check_correctness,
        n_correctness_iter,
    )

    get(ENV, "CUNUMERIC_BENCH_VERBOSE", "0") == "1" && println(
        "[$label] $name benchmark ($T_name) on $(N)x$(M) for $n_iter iterations " *
        "($n_warmup warmup) x $n_trial trials",
    )
    result = run_benchmark(
        benchmark, settings; mod=backend.mod, clock=backend.clock,
        synchronize=backend.synchronize,
    )
    println("[$label] Correctness: $(result.correctness)")
    @printf(
        "[%s] Mean time: %.5f ± %.5f ms (trial SD)\n",
        label, mean(result.times_ms), _std(result.times_ms),
    )
    @printf(
        "[%s] Mean throughput: %.5f ± %.5f %s (trial SD)\n",
        label, mean(result.gflops), _std(result.gflops), throughput_label(benchmark),
    )
    return save_result(result, gpus; mod=save_as)
end

run_array_worker()
