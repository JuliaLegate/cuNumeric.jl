# Shared sizing: explicit plans, no runtime probes and no mutable sizing cache.
struct PlannedRun
    spec::BenchmarkSpec
    model::Symbol
    N::Int
    M::Int
    memory::MemoryEstimate
    budget::Int
end

# Budget may be a single value applied to every spec or a per-spec lookup.
run_budget(budget::Integer, spec) = Int(budget)
run_budget(budget::AbstractDict, spec) = budget[spec]

function workspace_bound(raw, name, model)
    entry = get(get(raw, "workspace", Dict()), name, Dict())
    value = get(entry, string(model), nothing)
    value === nothing && return nothing
    value isa Integer && value >= 0 ||
        error("workspace.$name.$model must be a nonnegative byte count")
    return Int(value)
end

function contexts(spec, gs, raw)
    selected = ExecutionModel[]
    for id in spec.models
        model = execution_model(id)
        supports_run(model, spec.name, spec.gpus) && push!(selected, model)
    end
    isempty(selected) && error(
        "$(spec.name) on $(spec.gpus) GPU(s) has no implementation among selected models " *
        "$(join(string.(spec.models), ", "))",
    )
    return [
        MemoryContext(; model=model_id(model), fusion=spec.fusion,
            gpus=spec.gpus, steps=spec.n_warmup+spec.n_iter,
            workspace_bytes=workspace_bound(raw, spec.name, model_id(model))) for model in selected
    ]
end

function validate_spec(s)
    haskey(BENCHMARKS, s.name) || error("Unknown benchmark $(s.name)")
    s.gpus > 0 && s.cpus >= 0 || error("GPU count must be positive and CPU count nonnegative")
    s.n_iter > 0 && s.n_trial > 0 && s.n_warmup >= 0 || error("Invalid trial/iteration count")
    for hint in (s.N_hint, s.M_hint)
        hint === nothing || hint > 0 || error("Pinned dimensions must be positive")
    end
    if s.name in ("nas_ep", "nas_ft")
        label = s.name == "nas_ep" ? "NAS EP" : "NAS FT"
        s.T == "Float64" || error("$label requires Float64")
        s.n_iter == 1 || error("$label requires n_iter=1; use n_trial for repeated samples")
        s.autosize && error("$label uses fixed classes and cannot be autosized")
    end
end

function dimensions_at(s, baseline)
    !s.autosize && return Tuple(s.args)
    n, m = baseline
    b = build_benchmark(BENCHMARKS[s.name], parse_bench_type(s.T), n, m; s.kwargs...)
    result = estimate_scaling(b, s.gpus)
    result === nothing && error("$(s.name) cannot scale this baseline to $(s.gpus) GPUs")
    return result
end

function baseline_shape(s, k)
    B = BENCHMARKS[s.name]
    if B <: AbstractDMD
        return (something(s.N_hint, k), something(s.M_hint, DEFAULT_DMD_M))
    elseif B <: PoissonFFT
        return s.N_hint === nothing ? (k, something(s.M_hint, 1)) : (s.N_hint, k)
    elseif B <: AbstractMonteCarloIntegration || B <: AbstractTensorContraction ||
        B <: AbstractConjugateGradient
        s.M_hint === nothing || s.M_hint == 1 || error("$(s.name) requires M=1")
        return (something(s.N_hint, k), 1)
    else
        return (something(s.N_hint, k), something(s.M_hint, k))
    end
end

function candidate_runs(specs, gs, raw, baseline, budget)
    runs = PlannedRun[]
    seen = Set{Any}()
    for s in specs
        n, m = dimensions_at(s, baseline)
        b = build_benchmark(BENCHMARKS[s.name], parse_bench_type(s.T), n, m; s.kwargs...)
        for c in contexts(s, gs, raw)
            # Models without a fusion setting run only once.
            validate_model_kwargs(execution_model(c.model), s.name, s.kwargs)
            key = (s.name, s.T, s.gpus, s.cpus, n, m, c.model,
                uses_fusion(execution_model(c.model)) ? s.fusion : nothing,
                s.n_iter, s.n_warmup, s.n_trial, s.kwargs)
            key in seen && continue
            push!(seen, key)
            push!(runs, PlannedRun(s, c.model, n, m, memory_estimate(b, c), run_budget(budget, s)))
        end
    end
    # Multiple explicit blocks must not create misleading overlays.
    sizes = Dict{Tuple{String,Int},Tuple{Int,Int}}()
    for r in runs
        key = (r.spec.T, r.spec.gpus)
        previous = get!(sizes, key, (r.N, r.M))
        previous == (r.N, r.M) ||
            error("Comparison group has incompatible pinned dimensions at $(r.spec.gpus) GPUs")
    end
    return runs
end

function plan_runs(specs, gs, raw, groups, budget)
    isempty(specs) && error("No benchmarks selected")
    foreach(validate_spec, specs)
    group_for = Dict(member=>group for (group, members) in groups for member in members)
    group_members = Dict(groups)
    buckets = Dict{Any,Vector{BenchmarkSpec}}()
    order = Any[]
    for s in specs
        key = (get(group_for, s.name, s.name), s.T)
        if !haskey(buckets, key)
            buckets[key] = BenchmarkSpec[]
            push!(order, key)
        end
        push!(buckets[key], s)
    end
    planned = PlannedRun[]
    for key in order
        members = buckets[key]
        autos = filter(s->s.autosize, members)
        if isempty(autos)
            # A user who pins a size gets it; the run is kept and allowed to OOM
            # at runtime rather than blocked here on a conservative estimate.
            runs = candidate_runs(members, gs, raw, nothing, budget)
            for r in runs
                peak_bytes(r.memory) <= r.budget || @warn(
                    "Pinned size may exhaust GPU memory; run kept and allowed to OOM at runtime",
                    benchmark=r.spec.name, model=r.model, N=r.N, M=r.M,
                    peak=peak_bytes(r.memory), budget=r.budget)
            end
            append!(planned, runs)
            continue
        end
        length(autos)==length(members) ||
            error("Do not mix pinned and automatic sizes in comparison group $(key[1])")
        hints = unique((s.N_hint, s.M_hint) for s in members)
        length(hints)==1 || error("Incompatible size constraints in comparison group $(key[1])")
        s = first(members)
        B = BENCHMARKS[s.name]
        quantum = if B <: PoissonFFT && s.N_hint !== nothing
            1
        elseif B <: AbstractTensorContraction || B <: PoissonFFT
            2
        else
            8
        end
        minimum_n =
            B <: AbstractDMD ? max(something(s.M_hint, DEFAULT_DMD_M)*DMD_TALL_RATIO, 8) : quantum
        lo = cld(minimum_n, quantum)
        # At least one value of T must fit per candidate. Binary search uses
        # BigInt memory formulas and an Int-safe bound on scaled dimensions.
        hi = max(
            lo,
            Int(
                min(
                    maximum(run_budget(budget, m) for m in members)÷sizeof(parse_bench_type(s.T)),
                    typemax(Int)÷(8maximum(x.gpus for x in members)),
                ),
            )÷quantum,
        )
        make(k) = candidate_runs(members, gs, raw, baseline_shape(s, k*quantum), budget)
        fits(k) = all(r->peak_bytes(r.memory)<=r.budget, make(k))
        # Evaluate once before search to surface unsupported model errors.
        fits(lo) || error("Minimum problem does not fit in $(key[1])")
        best = largest_feasible(lo, hi, fits)
        best === nothing && error("No feasible size for $(key[1])")
        append!(planned, make(best))
    end
    return planned
end

function print_plan(runs, budget)
    println("Per-GPU budget shown per run; sizes are shared within each comparison group.")
    for r in runs
        m = r.memory
        println(
            "$(r.spec.name) / $(r.model) / $(r.spec.T) fusion=$(r.spec.fusion) GPUs=$(r.spec.gpus) N=$(r.N) M=$(r.M)"
        )
        println(
            "  initialization=$(m.initialization), iteration=$(m.iteration), workspace=$(m.workspace), peak=$(peak_bytes(m)) bytes; budget=$(r.budget) (mem_frac=$(r.spec.mem_frac))"
        )
        println("  $(m.explanation)")
    end
    limiting = runs[argmax([peak_bytes(r.memory) for r in runs])]
    return println(
        "Largest planned peak: $(limiting.spec.name) / $(limiting.model) / $(limiting.spec.gpus) GPUs"
    )
end
