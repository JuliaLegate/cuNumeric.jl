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

# quiet by default (banner + results only); -v/--verbose shows the plumbing
const VERBOSE_FLAGS = ("-v", "--verbose")
const VERBOSE = any(in(ARGS), VERBOSE_FLAGS)
const POSARGS = filter(a -> a ∉ VERBOSE_FLAGS, ARGS)

banner(msg) = println("\n", "="^128, "\n", msg, "\n", "="^128)

# `_accelerated` is a cuNumeric-only code-path variant (`@accelerate`).
cunumeric_only(name) = endswith(name, "_accelerated")

const LAST_FUSION_TOGGLE = Ref{Union{Nothing,Bool}}(nothing)

# dev CNPreferences && cuNumeric
function ensure_project_ready()
    Pkg.develop([
        Pkg.PackageSpec(; path=joinpath(@__DIR__, "..", "lib", "CNPreferences")),
        Pkg.PackageSpec(; path=joinpath(@__DIR__, "..")),
    ])
    return Pkg.instantiate()
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
    return error("could not resolve cupynumeric_jll version; set CUPYNUMERIC_ENV explicitly")
end

function dispatch(; gpus, cpus, name, T, N, M, n_iter, n_warmup, n_trial,
    fusion=true, cupynumeric=false, cudajl=false,
    check_correctness=false, n_correctness_iter=5)
    fstr = fusion ? "enabled" : "disabled"
    banner(
        "$(name): T=$(T) gpus=$(gpus) cpus=$(cpus) N=$(N) M=$(M) fusion=$(fstr) " *
        "n_iter=$(n_iter) n_warmup=$(n_warmup) n_trial=$(n_trial)",
    )

    # precompile in the orchestrator so the worker loads a warm cache quietly
    CNPreferences.set_broadcast_fusion!(fusion)
    if LAST_FUSION_TOGGLE[] != fusion
        VERBOSE && println("Precompiling cuNumeric (fusion=$(fstr))")
        Pkg.precompile("cuNumeric"; io=devnull)
        LAST_FUSION_TOGGLE[] = fusion
    end

    # each backend runs in its own worker process
    vflag = VERBOSE ? `--verbose` : ``
    args = `--gpus $gpus --cpus $cpus $name $T $N $M $n_iter $n_warmup $n_trial`
    # trailing: backend check_correctness n_correctness_iter
    corr_args = `$check_correctness $n_correctness_iter`
    cmds = [`bash $RUNNER $WORKER $vflag $args cunumeric $corr_args`]

    # comparison backends have no fusion knob, so run them once instead of per
    # fusion variant; the fused pass (the default) is that single run
    run_comparison_backends = fusion
    if run_comparison_backends
        # CUDA.jl is single-GPU only
        if cudajl && gpus == 1 && !cunumeric_only(name)
            push!(cmds, `bash $RUNNER $WORKER $vflag $args cudajl $corr_args`)
        end
        if cupynumeric && !cunumeric_only(name)
            push!(
                cmds,
                `bash $RUNNER $PY_WORKER $vflag --pyenv $(cupynumeric_env_name()) $args $corr_args`,
            )
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

function plot_all_results(config="benchmarks.toml")
    cfg = isabspath(config) ? config : joinpath(@__DIR__, config)
    plotter = joinpath(@__DIR__, "plot_results.jl")
    banner("Plotting results")
    try
        run(`$(Base.julia_cmd()) --project=$(@__DIR__) $plotter --config=$cfg`)
    catch e
        @error "Plotting failed; continuing." exception = e
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
            n_iter=spec.n_iter,
            n_warmup=spec.n_warmup,
            n_trial=spec.n_trial,
            cupynumeric=gs.cupynumeric,
            cudajl=spec.cuda,
            check_correctness=gs.check_correctness,
            n_correctness_iter=gs.n_correctness_iter,
        )
    end
    return plot_all_results(config)
end

ensure_project_ready()
using CNPreferences: CNPreferences
if isempty(POSARGS)
    run_all_benchmarks()
else # dispatch on args; inherit comparison-backend flags from benchmarks.toml
    gs, _ = parse_config(joinpath(@__DIR__, "benchmarks.toml"))
    dispatch(;
        gpus=parse(Int, POSARGS[1]),
        cpus=parse(Int, POSARGS[2]),
        name=POSARGS[3],
        T=POSARGS[4],
        N=parse(Int, POSARGS[5]),
        M=parse(Int, POSARGS[6]),
        n_iter=parse(Int, POSARGS[7]),
        n_warmup=parse(Int, POSARGS[8]),
        n_trial=parse(Int, POSARGS[9]),
        fusion=length(POSARGS) >= 10 ? parse_fusion(POSARGS[10]) : true,
        cupynumeric=gs.cupynumeric,
        cudajl=gs.cuda,
        check_correctness=length(POSARGS) >= 11 ? parse(Bool, POSARGS[11]) : gs.check_correctness,
        n_correctness_iter=length(POSARGS) >= 12 ? parse(Int, POSARGS[12]) : gs.n_correctness_iter,
    )
end
