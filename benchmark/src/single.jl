# single.jl: worker that runs exactly one benchmark under one backend. Launched by
# run_benchmark.sh (dispatched from run.jl), which sets LEGATE_CONFIG before julia starts.
# Args: <gpus> <name> <T> <N> <M> <n_iter> <n_warmup> <n_trial> <backend>
#       [check_correctness] [n_correctness_iter]
# backend is "cunumeric" or "cudajl"; run.jl launches one worker per backend.
# run.jl sets the compile-time fusion pref before launch; we read it back to label results.

using cuNumeric
using LinearAlgebra
using TensorOperations

length(ARGS) >= 9 || error(
    "single.jl args: <gpus> <name> <T> <N> <M> <n_iter> <n_warmup> <n_trial> <backend> " *
    "[check_correctness] [n_correctness_iter]",
)

const GPUS = parse(Int, ARGS[1])
const BACKEND_NAME = ARGS[9]
const CHECK_CORRECTNESS = length(ARGS) >= 10 ? parse(Bool, ARGS[10]) : false

# Timed CUDA.jl worker always needs CUDA. The 1-GPU cuNumeric worker loads it
# only for the tiny oracle check.
const NEED_CUDA =
    BACKEND_NAME == "cudajl" ||
    (BACKEND_NAME == "cunumeric" && CHECK_CORRECTNESS && GPUS == 1)

if NEED_CUDA
    using CUDA
    using AbstractFFTs
    using cuTENSOR
end

include("core.jl")
include_benchmarks()

# Resolve a TOML type string like "Float32" to the actual Julia type.
parse_type(s) = getfield(Base, Symbol(s))::DataType

# One worker process, one backend. Clock functions have distinct types, so do
# not stash them in a Dict inferred from the cuNumeric entry.
function backend_entry(name)
    name == "cunumeric" && return (
        mod=cuNumeric, label="cuNumeric", save_as="cunumeric",
        clock=get_time_microseconds,
    )
    if name == "cudajl"
        cuda_clock() = (CUDA.synchronize(; blocking=true); time_ns() / 1e3)
        return (mod=CUDA, label="CUDA.jl", save_as="CUDA.jl", clock=cuda_clock)
    end
    return error("Unknown backend '$(name)'. Known: cunumeric, cudajl")
end
const BACKEND = backend_entry(BACKEND_NAME)

function run_single(
    gpus, name, T_str, N, M, n_iter, n_warmup, n_trial, backend;
    check_correctness=false, n_correctness_iter=5,
)
    haskey(BENCHMARKS, name) || error(
        "No benchmark registered for '$(name)'. Known: $(join(sort(collect(keys(BENCHMARKS))), ", "))"
    )
    bk = BACKEND

    T = parse_type(T_str)
    b = build_benchmark(BENCHMARKS[name], T, N, M)

    # unfused cuNumeric runs land in their own CSV so they stay a distinct series
    fused = cuNumeric.FUSE_BROADCAST_EXPRS
    default_save_as = fused ? bk.save_as : "$(bk.save_as)_nofusion"
    default_label = fused ? bk.label : "$(bk.label) (no fusion)"
    save_as = benchmark_backend_save_as(b, backend, default_save_as)
    label = benchmark_backend_label(b, backend, default_label)
    gs = GlobalSettings(;
        n_warmup=n_warmup,
        n_iter=n_iter,
        n_trial=n_trial,
        n_gpu=gpus,
        check_correctness=check_correctness,
        n_correctness_iter=n_correctness_iter,
    )

    println(
        "[$(label)] $(name) benchmark ($(T)) on $(N)x$(M) for $(n_iter) " *
        "iterations ($(n_warmup) warmup) x $(n_trial) trials",
    )
    br = run_benchmark(b, gs; mod=bk.mod, clock=bk.clock)
    @printf("[%s] Mean Run Time: %.5f ± %.5f ms\n", label, mean(br.times_ms), _std(br.times_ms))
    @printf("[%s] FLOPS: %.5f ± %.5f GFLOPS\n", label, mean(br.gflops), _std(br.gflops))
    println("[$(label)] Correctness: $(br.correctness)")
    return save_result(br, gpus; mod=save_as)
end

bench_name = ARGS[2]
T_str = ARGS[3]
N = parse(Int, ARGS[4])
M = parse(Int, ARGS[5])
n_iter = parse(Int, ARGS[6])
n_warmup = parse(Int, ARGS[7])
n_trial = parse(Int, ARGS[8])
n_correctness_iter = length(ARGS) >= 11 ? parse(Int, ARGS[11]) : 5
run_single(
    GPUS, bench_name, T_str, N, M, n_iter, n_warmup, n_trial, BACKEND_NAME;
    check_correctness=CHECK_CORRECTNESS, n_correctness_iter=n_correctness_iter,
)
