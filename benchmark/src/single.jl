# single.jl: worker that runs exactly one benchmark under one backend. Launched by
# run_benchmark.sh (dispatched from run.jl), which sets LEGATE_CONFIG before julia starts.
# Args: <gpus> <name> <T> <N> <M> <n_iter> <n_warmup> <n_trial> <backend>
#       [check_correctness] [n_correctness_iter]
# backend is "cunumeric" or "cudajl"; run.jl launches one worker per backend.
# run.jl sets the compile-time fusion pref before launch; we read it back to label results.

using cuNumeric
using CUDACore
using LinearAlgebra

include("core.jl")
const BENCHMARK_DIR = joinpath(@__DIR__, "benchmarks")
include.(filter(contains(r".jl$"), readdir(BENCHMARK_DIR; join=true)))

# Resolve a TOML type string like "Float32" to the actual Julia type.
parse_type(s) = getfield(Base, Symbol(s))::DataType

# mod runs the kernels; label tags stdout; save_as names the results CSV.
const BACKENDS = Dict(
    "cunumeric" => (mod=cuNumeric, label="cuNumeric", save_as="cunumeric"),
    "cudajl" => (mod=CUDACore, label="CUDA.jl", save_as="CUDA.jl"),
)

function run_single(
    gpus, name, T_str, N, M, n_iter, n_warmup, n_trial, backend;
    check_correctness=false, n_correctness_iter=5,
)
    haskey(BENCHMARKS, name) || error(
        "No benchmark registered for '$(name)'. Known: $(join(sort(collect(keys(BENCHMARKS))), ", "))"
    )
    haskey(BACKENDS, backend) || error(
        "Unknown backend '$(backend)'. Known: $(join(sort(collect(keys(BACKENDS))), ", "))"
    )
    bk = BACKENDS[backend]

    # unfused cuNumeric runs land in their own CSV so they stay a distinct series
    fused = cuNumeric.FUSE_BROADCAST_EXPRS
    save_as = fused ? bk.save_as : "$(bk.save_as)_nofusion"
    label = fused ? bk.label : "$(bk.label) (no fusion)"

    T = parse_type(T_str)
    b = build_benchmark(BENCHMARKS[name], T, N, M)
    gs = GlobalSettings(;
        n_warmup=n_warmup,
        n_iter=n_iter,
        n_trial=n_trial,
        check_correctness=check_correctness,
        n_correctness_iter=n_correctness_iter,
    )

    println(
        "[$(label)] $(name) benchmark ($(T)) on $(N)x$(M) for $(n_iter) " *
        "iterations ($(n_warmup) warmup) x $(n_trial) trials",
    )
    br = run_benchmark(b, gs; mod=bk.mod)
    @printf("[%s] Mean Run Time: %.5f ± %.5f ms\n", label, mean(br.times_ms), _std(br.times_ms))
    @printf("[%s] FLOPS: %.5f ± %.5f GFLOPS\n", label, mean(br.gflops), _std(br.gflops))
    println("[$(label)] Correctness: $(br.correctness)")
    return save_result(br, gpus; mod=save_as)
end

gpus = parse(Int, ARGS[1])
bench_name = ARGS[2]
T_str = ARGS[3]
N = parse(Int, ARGS[4])
M = parse(Int, ARGS[5])
n_iter = parse(Int, ARGS[6])
n_warmup = parse(Int, ARGS[7])
n_trial = parse(Int, ARGS[8])
backend = ARGS[9]
check_correctness = length(ARGS) >= 10 ? parse(Bool, ARGS[10]) : false
n_correctness_iter = length(ARGS) >= 11 ? parse(Int, ARGS[11]) : 5
run_single(
    gpus, bench_name, T_str, N, M, n_iter, n_warmup, n_trial, backend;
    check_correctness=check_correctness, n_correctness_iter=n_correctness_iter,
)
