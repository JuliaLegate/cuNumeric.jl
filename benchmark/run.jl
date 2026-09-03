# run.jl: orchestrator. Builds one run_benchmark.sh command per benchmark and
# dispatches it; the script sets LEGATE_CONFIG (from --gpus/--cpus) before
# launching the worker (single.jl) that actually runs the benchmark.
#   no args   -> one command per benchmarks.toml entry
#   with args -> one command from <gpus> <cpus> <name> <T> <N> <M> <iter> <warmup> <trial> [fusion]
# N/M may be integers or "auto".

# Orchestrator stays off the GPU: it includes kernel files only for types,
# total_space, and estimate_scaling. The worker loads cuNumeric.

using Pkg

# Instantiated before kernel files are included (`using TensorOperations`).
function ensure_project_ready()
    Pkg.develop([
        Pkg.PackageSpec(; path=joinpath(@__DIR__, "..", "lib", "CNPreferences")),
        Pkg.PackageSpec(; path=joinpath(@__DIR__, "..")),
    ])
    return Pkg.instantiate()
end
ensure_project_ready()

include("src/core.jl")
include_benchmarks()
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

function flop_count(name, T_str, N, M)
    B = BENCHMARKS[name]
    T = parse_bench_type(T_str)
    return total_flops(build_benchmark(B, T, N, M))
end

function cached_autosize(spec, gs, cache)
    key = (spec.name, spec.T, spec.N_hint, spec.M_hint)
    fit = get!(cache, key) do
        B = BENCHMARKS[spec.name]
        T = parse_bench_type(spec.T)
        budget, frac = autosize_budget(gs.mem_frac)
        N1, M1 = fit_one_gpu(B, T; budget, spec.N_hint, spec.M_hint)
        (; N1, M1, budget, frac)
    end
    scaled = estimate_scaling(
        build_benchmark(BENCHMARKS[spec.name], parse_bench_type(spec.T), fit.N1, fit.M1),
        spec.gpus,
    )
    scaled === nothing && return nothing
    N, M = scaled
    return (; N, M, fit.N1, fit.M1, fit.budget, fit.frac)
end

function parse_cli_size(s)
    is_auto_size(s) && return :auto
    return parse(Int, s)
end

function dispatch(; gpus, cpus, name, T, N, M, n_iter, n_warmup, n_trial,
    fusion=true, cupynumeric=false, cudajl=false,
    check_correctness=false, n_correctness_iter=5)
    fstr = fusion ? "enabled" : "disabled"
    flops = flop_count(name, T, N, M)
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
    # trailing: backend check_correctness n_correctness_iter flops
    # flops is computed here so Python does not duplicate Julia formulas
    corr_args = `$check_correctness $n_correctness_iter $flops`
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
    cache = Dict{Any,Any}()
    for spec in specs
        N, M = spec.args
        if spec.autosize
            r = cached_autosize(spec, gs, cache)
            if r === nothing
                println(
                    "Skipping $(spec.name) on $(spec.gpus) GPUs: 1-GPU size is not in " *
                    "the tall-skinny DMD regime (N ≥ $(DMD_TALL_RATIO)×M).",
                )
                continue
            end
            N, M = r.N, r.M
            println(
                "autosize $(spec.name): 1-GPU ($(r.N1), $(r.M1)) → $(spec.gpus) GPU(s) " *
                "($(N), $(M)); budget=$(r.budget) bytes (mem_frac=$(r.frac))",
            )
        end
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

using CNPreferences: CNPreferences
if isempty(POSARGS)
    run_all_benchmarks()
else # dispatch on args; inherit comparison-backend flags from benchmarks.toml
    gs, _ = parse_config(joinpath(@__DIR__, "benchmarks.toml"))
    Narg = parse_cli_size(POSARGS[5])
    Marg = parse_cli_size(POSARGS[6])
    name = POSARGS[3]
    T = POSARGS[4]
    gpus = parse(Int, POSARGS[1])
    N = 0
    M = 0
    if Narg === :auto || Marg === :auto
        r = resolve_autosize(
            name, T, gpus;
            mem_frac=gs.mem_frac,
            N_hint=(Narg === :auto ? nothing : Narg),
            M_hint=(Marg === :auto ? nothing : Marg),
        )
        if r === nothing
            println(
                "Skipping $(name) on $(gpus) GPUs: 1-GPU size is not in " *
                "the tall-skinny DMD regime (N ≥ $(DMD_TALL_RATIO)×M).",
            )
            exit(0)
        end
        N, M = r.N, r.M
        println(
            "autosize $(name): 1-GPU ($(r.N1), $(r.M1)) → $(gpus) GPU(s) ($(N), $(M)); " *
            "budget=$(r.budget) bytes (mem_frac=$(r.frac))",
        )
    else
        N, M = Narg, Marg
    end
    dispatch(;
        gpus=gpus,
        cpus=parse(Int, POSARGS[2]),
        name=name,
        T=T,
        N=N, M=M,
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
