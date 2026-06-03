# single.jl: worker that runs exactly one benchmark. Launched by run_benchmark.sh
# (dispatched from run.jl), which sets LEGATE_CONFIG in the env before julia starts.
# Args: <gpus> <name> <T> <N> <M> <n_iter> <n_warmup> <n_trial> <variant>

using cuNumeric
using CUDACore
using LinearAlgebra

include("core.jl")
const BENCHMARK_DIR = joinpath(@__DIR__, "benchmarks")
include.(filter(contains(r".jl$"), readdir(BENCHMARK_DIR; join=true)))

# Resolve a TOML type string like "Float32" to the actual Julia type.
parse_type(s) = getfield(Base, Symbol(s))::DataType

function run_single(gpus, name, T_str, N, M, n_iter, n_warmup, n_trial)
    haskey(BENCHMARKS, name) || error(
        "No benchmark registered for '$(name)'. Known: $(join(sort(collect(keys(BENCHMARKS))), ", "))"
    )
    T = parse_type(T_str)
    b = build_benchmark(BENCHMARKS[name], T, N, M)
    gs = GlobalSettings(; n_warmup=n_warmup, n_iter=n_iter, n_trial=n_trial)

    println(
        "[cuNumeric] $(name) benchmark ($(T)) on $(N)x$(M) for $(n_iter) " *
        "iterations ($(n_warmup) warmup) x $(n_trial) trials",
    )
    br = run_benchmark(b, gs)
    @printf("[cuNumeric] Mean Run Time: %.5f ± %.5f ms\n", mean(br.times_ms), _std(br.times_ms))
    @printf("[cuNumeric] FLOPS: %.5f ± %.5f GFLOPS\n", mean(br.gflops), _std(br.gflops))

    save_result(br, gpus)

    # Run CUDA.jl benchmark
    if gpus == 1
        println(
            "[CUDA.jl] $(name) benchmark ($(T)) on $(N)x$(M) for $(n_iter) " *
            "iterations ($(n_warmup) warmup) x $(n_trial) trials",
        )

        br = run_benchmark(b, gs; mod=CUDACore)

        @printf("[CUDA.jl] Mean Run Time: %.5f ± %.5f ms\n", mean(br.times_ms), _std(br.times_ms))
        @printf("[CUDA.jl] FLOPS: %.5f ± %.5f GFLOPS\n", mean(br.gflops), _std(br.gflops))

        save_result(br, gpus; mod="CUDA.jl")
    end
end

gpus = parse(Int, ARGS[1])
bench_name = ARGS[2]
T_str = ARGS[3]
N = parse(Int, ARGS[4])
M = parse(Int, ARGS[5])
n_iter = parse(Int, ARGS[6])
n_warmup = parse(Int, ARGS[7])
n_trial = parse(Int, ARGS[8])
run_single(gpus, bench_name, T_str, N, M, n_iter, n_warmup, n_trial)
