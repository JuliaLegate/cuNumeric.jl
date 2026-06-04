# run.jl: orchestrator. Builds one run_benchmark.sh command per benchmark and
# dispatches it; the script sets LEGATE_CONFIG (from --gpus/--cpus) before
# launching the worker (single.jl) that actually runs the benchmark.
#   no args   -> one command per benchmarks.toml entry
#   with args -> one command from <gpus> <cpus> <name> <T> <N> <M> <iter> <warmup> <trial> [fusion]

# Orchestrator stays off the GPU: it only needs GlobalSettings + parse_config,
# both cuNumeric-free. The worker (single.jl) loads cuNumeric and the kernels.

using Pkg

include("src/core.jl")
include("src/parse_benchmarks.jl")

const RUNNER = joinpath(@__DIR__, "run_benchmark.sh")
const WORKER = joinpath(@__DIR__, "src/single.jl")
const PY_WORKER = joinpath(@__DIR__, "src_py/single.py")

banner(msg) = println("\n", "="^128, "\n", msg, "\n", "="^128)

# `_lifetimes` is a cuNumeric-only code-path variant (@analyze_lifetimes)
cunumeric_only(name) = endswith(name, "_lifetimes")

# dev CNPreferences && cuNumeric
function ensure_project_ready()
    Pkg.develop([
        Pkg.PackageSpec(; path=joinpath(@__DIR__, "..", "lib", "CNPreferences")),
        Pkg.PackageSpec(; path=joinpath(@__DIR__, "..")),
    ])
    Pkg.instantiate()
end

# default env name mirrors install_cupynumeric.sh: cupynumeric-bench-<major>.<minor>
# CUPYNUMERIC_ENV overrides it.
function cupynumeric_env_name()
    haskey(ENV, "CUPYNUMERIC_ENV") && return ENV["CUPYNUMERIC_ENV"]
    for (_, info) in Pkg.dependencies()
        info.name == "cupynumeric_jll" || continue
        info.version === nothing && continue
        return "cupynumeric-bench-$(info.version.major).$(info.version.minor)"
    end
    error("could not resolve cupynumeric_jll version; set CUPYNUMERIC_ENV explicitly")
end

function dispatch(; gpus, cpus, name, T, N, M, n_iter, n_warmup, n_trial,
    fusion=true, cupynumeric=false, cudajl=false)
    fstr = fusion ? "enabled" : "disabled"
    banner(
        "$(name): T=$(T) gpus=$(gpus) cpus=$(cpus) N=$(N) M=$(M) fusion=$(fstr) " *
        "n_iter=$(n_iter) n_warmup=$(n_warmup) n_trial=$(n_trial)",
    )

    # set the compile-time fusion pref before the worker loads cuNumeric
    # cuNumeric-only, so comparison backends run once under the default (fused) config
    CNPreferences.set_broadcast_fusion!(fusion)

    # each backend runs in its own worker process
    args = `--gpus $gpus --cpus $cpus $name $T $N $M $n_iter $n_warmup $n_trial`
    cmds = [`bash $RUNNER $WORKER $args cunumeric`]

    # comparison backends have no fusion knob, so run them once instead of per
    # fusion variant; the fused pass (the default) is that single run
    run_comparison_backends = fusion
    if run_comparison_backends
        # CUDA.jl is single-GPU only
        if cudajl && gpus == 1 && !cunumeric_only(name)
            push!(cmds, `bash $RUNNER $WORKER $args cudajl`)
        end
        if cupynumeric && !cunumeric_only(name)
            push!(cmds, `bash $RUNNER $PY_WORKER --pyenv $(cupynumeric_env_name()) $args`)
        end
    end

    for cmd in cmds
        try
            run(cmd)
        catch e
            @error "Benchmark '$(name)' failed; continuing." exception = e
        end
    end
end

function run_all_benchmarks(config="benchmarks.toml")
    gs, specs = parse_config(joinpath(@__DIR__, config))
    for spec in specs
        N, M = spec.args
        dispatch(;
            gpus=spec.gpus,
            cpus=spec.cpus,
            name=spec.name,
            T=spec.T,
            N=N, M=M,
            fusion=spec.fusion,
            n_iter=gs.n_iter,
            n_warmup=gs.n_warmup,
            n_trial=gs.n_trial,
            cupynumeric=gs.cupynumeric,
            cudajl=gs.cuda,
        )
    end
end

ensure_project_ready()
using CNPreferences: CNPreferences
if isempty(ARGS)
    run_all_benchmarks()
else # dispatch on args
    dispatch(;
        gpus=parse(Int, ARGS[1]),
        cpus=parse(Int, ARGS[2]),
        name=ARGS[3],
        T=ARGS[4],
        N=parse(Int, ARGS[5]),
        M=parse(Int, ARGS[6]),
        n_iter=parse(Int, ARGS[7]),
        n_warmup=parse(Int, ARGS[8]),
        n_trial=parse(Int, ARGS[9]),
        fusion=length(ARGS) >= 10 ? parse_fusion(ARGS[10]) : true,
    )
end
