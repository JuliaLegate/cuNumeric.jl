# run.jl: orchestrator. Builds one run_benchmark.sh command per benchmark and
# dispatches it; the script sets LEGATE_CONFIG (from --gpus/--cpus) before
# launching the worker (single.jl) that actually runs the benchmark.
#   no args   -> one command per benchmarks.toml entry
#   with args -> one command from <gpus> <cpus> <name> <T> <N> <M> <iter> <warmup> <trial> [variant]

# Orchestrator stays off the GPU: it only needs GlobalSettings + parse_config,
# both cuNumeric-free. The worker (single.jl) loads cuNumeric and the kernels.
include("src/core.jl")
include("src/parse_benchmarks.jl")

const RUNNER = joinpath(@__DIR__, "run_benchmark.sh")
const WORKER = joinpath(@__DIR__, "src/single.jl")

banner(msg) = println("\n", "="^128, "\n", msg, "\n", "="^128)

function dispatch(; gpus, cpus, name, T, variant, N, M, n_iter, n_warmup, n_trial)
    # Name validity is checked in the worker (single.jl), which owns the registry.
    banner(
        "$(name) [$(variant)]: T=$(T) gpus=$(gpus) cpus=$(cpus) N=$(N) M=$(M) " *
        "n_iter=$(n_iter) n_warmup=$(n_warmup) n_trial=$(n_trial)",
    )

    cmd = `bash $RUNNER $WORKER --gpus $gpus --cpus $cpus $name $T $N $M $n_iter $n_warmup $n_trial $variant`
    try
        run(cmd)
    catch e
        @error "Benchmark '$(name)' failed; continuing." exception = e
    end
end

function run_all_benchmarks(config="benchmarks.toml")
    gs, specs = parse_config(joinpath(@__DIR__, config))
    for spec in specs
        N, M = spec.args
        dispatch(;
            gpus=spec.gpus, cpus=spec.cpus, name=spec.name, T=spec.T,
            variant=spec.variant, N=N, M=M,
            n_iter=gs.n_iter, n_warmup=gs.n_warmup, n_trial=gs.n_trial,
        )
    end
end

if isempty(ARGS)
    run_all_benchmarks()
else # dispatch on args
    dispatch(;
        gpus=parse(Int, ARGS[1]), cpus=parse(Int, ARGS[2]), name=ARGS[3], T=ARGS[4],
        N=parse(Int, ARGS[5]), M=parse(Int, ARGS[6]),
        n_iter=parse(Int, ARGS[7]), n_warmup=parse(Int, ARGS[8]), n_trial=parse(Int, ARGS[9]),
        variant=(length(ARGS) >= 10 ? ARGS[10] : "baseline"),
    )
end
