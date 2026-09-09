using Dates, Pkg

function cli_options(args)
    config = joinpath(@__DIR__,"..","benchmarks.toml")
    only = nothing
    fusion = nothing
    dry = false
    verbose = false
    positional = String[]
    for arg in args
        if startswith(arg,"--only=")
            only = split(arg,'=';limit=2)[2]
        elseif startswith(arg,"--config=")
            config = abspath(split(arg,'=';limit=2)[2])
        elseif startswith(arg,"--fusion=")
            value = split(arg,'=';limit=2)[2]
            fusion = value == "both" ? [true,false] : [parse_fusion(value)]
        elseif arg == "--dry-run"
            dry = true
        elseif arg in ("-v","--verbose")
            verbose = true
        elseif startswith(arg,"--")
            error("Unknown option $arg")
        else
            push!(positional,arg)
        end
    end
    return (;config,only,fusion,dry,verbose,positional)
end

function positional_spec(p,gs)
    9 <= length(p) <= 12 || error("Expected <gpus> <cpus> <name> <T> <N> <M> <iter> <warmup> <trial> [fusion] [check_correctness] [correctness_iter]")
    n = is_auto_size(p[5]) ? nothing : parse(Int,p[5])
    m = is_auto_size(p[6]) ? nothing : parse(Int,p[6])
    auto = n === nothing || m === nothing
    return BenchmarkSpec(p[3],p[4],parse(Int,p[1]),parse(Int,p[2]),
        length(p)>=10 ? parse_fusion(p[10]) : true,gs.cuda,
        parse(Int,p[8]),parse(Int,p[7]),parse(Int,p[9]),
        auto ? [0,0] : [n,m],auto,n,m)
end

function plan_manifest(runs,budget,raw)
    versions = Dict(info.name=>string(info.version) for info in values(Pkg.dependencies()) if info.version !== nothing)
    return Dict("status"=>"running","budget_bytes"=>budget,"julia"=>string(VERSION),
        "versions"=>versions,"config"=>raw,"runs"=>[Dict{String,Any}(
            "name"=>r.spec.name,"T"=>r.spec.T,"backend"=>string(r.backend),
            "fusion"=>r.spec.fusion,"gpus"=>r.spec.gpus,"cpus"=>r.spec.cpus,
            "N"=>r.N,"M"=>r.M,"n_iter"=>r.spec.n_iter,"n_warmup"=>r.spec.n_warmup,
            "n_trial"=>r.spec.n_trial,"initialization_bytes"=>string(r.memory.initialization),
            "iteration_bytes"=>string(r.memory.iteration),"workspace_bytes"=>string(r.memory.workspace),
            "memory_explanation"=>r.memory.explanation,"status"=>"pending") for r in runs])
end

function prepare_backend(fusion,verbose)
    println("Setting fusion=$fusion and precompiling cuNumeric...")
    CNPreferences.set_broadcast_fusion!(fusion)
    Pkg.precompile("cuNumeric";io=verbose ? stderr : devnull)
end

function preflight_backends(runs;env=ENV,which=Sys.which,check=success)
    any(r.backend==:cupynumeric for r in runs) || return nothing
    conda = get(env,"CUNUMERIC_BENCH_CONDA",get(env,"CONDA_EXE","conda"))
    executable = which(conda)
    executable === nothing && error(
        "cuPyNumeric is enabled, but conda is not available to the worker. " *
        "Add conda to PATH or set CUNUMERIC_BENCH_CONDA to its executable path; " *
        "then run bash install_cupynumeric.sh. No benchmarks have been started.",
    )
    name = get(env,"CUPYNUMERIC_ENV",nothing)
    name === nothing && (name = cupynumeric_env_name())
    code = "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('cupynumeric') else 1)"
    check(`$executable run --no-capture-output -n $name python -c $code`) || error(
        "Conda environment '$name' is unavailable or lacks cupynumeric. " *
        "Run bash install_cupynumeric.sh, or set CUPYNUMERIC_ENV to an existing environment. " *
        "No benchmarks have been started.",
    )
    return nothing
end

function execute_plan(runs,gs,opts,budget,raw;launch=run,prepare=prepare_backend,
    results_root=normpath(joinpath(@__DIR__,"..","results")),preflight=preflight_backends)
    preflight(runs)
    root = normpath(joinpath(@__DIR__,".."))
    mkpath(results_root)
    dir = mktempdir(results_root;prefix=Dates.format(now(),"yyyymmdd-HHMMSS")*"-",cleanup=false)
    manifest = plan_manifest(runs,budget,raw)
    manifest_path = joinpath(dir,"manifest.toml")
    save_manifest() = open(io->TOML.print(io,manifest),manifest_path,"w")
    save_manifest()
    last_fusion = nothing
    failed = false
    for (i,r) in enumerate(runs)
        s = r.spec
        println("\n[$i/$(length(runs))] $(s.name) / $(r.backend), $(s.gpus) GPUs, $(r.N) × $(r.M)")
        try
            if r.backend == :cunumeric && last_fusion != s.fusion
                prepare(s.fusion,opts.verbose)
                last_fusion = s.fusion
            end
            b = build_benchmark(BENCHMARKS[s.name],parse_bench_type(s.T),r.N,r.M)
            p = opts.positional
            correctness = length(p)>=11 ? parse(Bool,p[11]) : gs.check_correctness
            correct_iters = length(p)>=12 ? parse(Int,p[12]) : gs.n_correctness_iter
            args = `--gpus $(s.gpus) --cpus $(s.cpus) $(s.name) $(s.T) $(r.N) $(r.M) $(s.n_iter) $(s.n_warmup) $(s.n_trial)`
            corr = `$correctness $correct_iters $(total_flops(b))`
            runner = joinpath(root,"run_benchmark.sh")
            verbose = opts.verbose ? `--verbose` : ``
            cmd = if r.backend == :cupynumeric
                worker = joinpath(root,"src_py","single.py")
                `bash $runner $worker $verbose --pyenv $(cupynumeric_env_name()) $args $corr`
            else
                worker = joinpath(root,"src","single.jl")
                `bash $runner $worker $verbose $args $(string(r.backend)) $corr`
            end
            results = joinpath(dir,s.T)
            launch(addenv(Cmd(cmd;dir=root),"CUNUMERIC_BENCH_RESULTS_DIR"=>results,
                "CUNUMERIC_BENCH_JULIA"=>joinpath(Sys.BINDIR,Base.julia_exename())))
            manifest["runs"][i]["status"] = "complete"
        catch e
            failed = true
            manifest["runs"][i]["status"] = "failed"
            # ProcessFailedException prints inherited environment variables.
            # Report exit codes without copying that environment into logs.
            message = e isa ProcessFailedException ?
                "Worker exited with code(s) " * join((p.exitcode for p in e.procs),", ") : sprint(showerror,e)
            manifest["runs"][i]["error"] = message
            @error "Worker failed; no size retry. Continuing independent configurations." reason=message
        end
        save_manifest()
    end
    manifest["status"] = failed ? "incomplete" : "complete"
    if isempty(opts.positional)
        for T in unique(r.spec.T for r in runs)
            try
                plotter = joinpath(root,"plot_results.jl")
                results = joinpath(dir,T)
                out = joinpath(root,"plots",basename(dir),T)
                suffix = failed ? "_incomplete" : ""
                launch(`$(Base.julia_cmd()) --project=$root $plotter $results --config=$(opts.config) --out=$out --suffix=$suffix`)
            catch e
                failed = true
                manifest["status"] = "incomplete"
                @error "Plotting failed" exception=e
            end
        end
    end
    save_manifest()
    println("Results: $dir ($(manifest["status"]))")
    return failed ? 1 : 0
end

function main(args=ARGS;budget_provider=selected_gpu_budget,executor=execute_plan)
    opts = cli_options(args)
    gs,specs = parse_config(opts.config;only=opts.only,fusion_override=opts.fusion)
    if !isempty(opts.positional)
        opts.only === nothing && opts.fusion === nothing || error("Do not mix positional runs with sweep filters")
        specs = [positional_spec(opts.positional,gs)]
    end
    isempty(specs) && error("No benchmarks selected")
    raw = TOML.parsefile(opts.config)
    budget,_ = budget_provider(gs.mem_frac,maximum(s.gpus for s in specs))
    runs = plan_runs(specs,gs,raw,parse_plot_groups(opts.config),budget)
    print_plan(runs,budget)
    opts.dry && return 0
    return executor(runs,gs,opts,budget,raw)
end
