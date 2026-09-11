# Shared sizing: explicit plans, no runtime probes and no mutable sizing cache.
struct PlannedRun
    spec::BenchmarkSpec
    backend::Symbol
    N::Int
    M::Int
    memory::MemoryEstimate
end

function workspace_bound(raw, name, backend)
    entry = get(get(raw, "workspace", Dict()), name, Dict())
    value = get(entry, string(backend), nothing)
    value === nothing && return nothing
    value isa Integer && value >= 0 || error("workspace.$name.$backend must be a nonnegative byte count")
    return Int(value)
end

function contexts(spec, gs, raw)
    backends = Symbol[:cunumeric]
    if !endswith(spec.name, "_accelerated")
        spec.cuda && spec.gpus == 1 && push!(backends, :cudajl)
        gs.cupynumeric && push!(backends, :cupynumeric)
    end
    return [MemoryContext(; backend, fusion=spec.fusion, gpus=spec.gpus,steps=spec.n_warmup+spec.n_iter,
        workspace_bytes=workspace_bound(raw, spec.name, backend)) for backend in backends]
end

function validate_spec(s)
    haskey(BENCHMARKS,s.name) || error("Unknown benchmark $(s.name)")
    s.gpus > 0 && s.cpus >= 0 || error("GPU count must be positive and CPU count nonnegative")
    s.n_iter > 0 && s.n_trial > 0 && s.n_warmup >= 0 || error("Invalid trial/iteration count")
    for hint in (s.N_hint,s.M_hint)
        hint === nothing || hint > 0 || error("Pinned dimensions must be positive")
    end
end

function dimensions_at(s, baseline)
    !s.autosize && return Tuple(s.args)
    n,m = baseline
    b = build_benchmark(BENCHMARKS[s.name],parse_bench_type(s.T),n,m)
    result = estimate_scaling(b,s.gpus)
    result === nothing && error("$(s.name) cannot scale this baseline to $(s.gpus) GPUs")
    return result
end

function baseline_shape(s, k)
    B = BENCHMARKS[s.name]
    if B <: AbstractDMD
        return (something(s.N_hint,k), something(s.M_hint,DEFAULT_DMD_M))
    elseif B <: PoissonFFT
        return s.N_hint === nothing ? (k,something(s.M_hint,1)) : (s.N_hint,k)
    elseif B <: MonteCarloIntegration || B <: AbstractTensorContraction
        s.M_hint === nothing || s.M_hint == 1 || error("$(s.name) requires M=1")
        return (something(s.N_hint,k),1)
    else
        return (something(s.N_hint,k),something(s.M_hint,k))
    end
end

function candidate_runs(specs,gs,raw,baseline)
    runs = PlannedRun[]
    seen = Set{Any}()
    for s in specs
        n,m = dimensions_at(s,baseline)
        b = build_benchmark(BENCHMARKS[s.name],parse_bench_type(s.T),n,m)
        for c in contexts(s,gs,raw)
            # Comparison backends have no fusion setting and run only once.
            key = (s.name,s.T,s.gpus,s.cpus,n,m,c.backend,
                c.backend == :cunumeric ? s.fusion : nothing,s.n_iter,s.n_warmup,s.n_trial)
            key in seen && continue
            push!(seen,key)
            push!(runs,PlannedRun(s,c.backend,n,m,memory_estimate(b,c)))
        end
    end
    # Multiple explicit blocks must not create misleading overlays.
    sizes = Dict{Tuple{String,Int},Tuple{Int,Int}}()
    for r in runs
        key = (r.spec.T,r.spec.gpus)
        previous = get!(sizes,key,(r.N,r.M))
        previous == (r.N,r.M) || error("Comparison group has incompatible pinned dimensions at $(r.spec.gpus) GPUs")
    end
    return runs
end

function plan_runs(specs,gs,raw,groups,budget::Integer)
    isempty(specs) && error("No benchmarks selected")
    foreach(validate_spec,specs)
    group_for = Dict(member=>group for (group,members) in groups for member in members)
    buckets = Dict{Any,Vector{BenchmarkSpec}}()
    order = Any[]
    for s in specs
        key = (get(group_for,s.name,s.name),s.T)
        if !haskey(buckets,key)
            buckets[key] = BenchmarkSpec[]
            push!(order,key)
        end
        push!(buckets[key],s)
    end
    planned = PlannedRun[]
    for key in order
        members = buckets[key]
        autos = filter(s->s.autosize,members)
        if isempty(autos)
            runs = candidate_runs(members,gs,raw,nothing)
            all(r->peak_bytes(r.memory)<=budget,runs) || error("Pinned size exceeds memory budget in $(key[1])")
            append!(planned,runs)
            continue
        end
        length(autos)==length(members) || error("Do not mix pinned and automatic sizes in comparison group $(key[1])")
        hints = unique((s.N_hint,s.M_hint) for s in members)
        length(hints)==1 || error("Incompatible size constraints in comparison group $(key[1])")
        s = first(members)
        B = BENCHMARKS[s.name]
        quantum = B <: PoissonFFT && s.N_hint !== nothing ? 1 : B <: AbstractTensorContraction || B <: PoissonFFT ? 2 : 8
        minimum_n = B <: AbstractDMD ? max(something(s.M_hint,DEFAULT_DMD_M)*DMD_TALL_RATIO,8) : quantum
        lo = cld(minimum_n,quantum)
        # At least one value of T must fit per candidate. Binary search uses
        # BigInt memory formulas and an Int-safe bound on scaled dimensions.
        hi = max(lo,Int(min(budget÷sizeof(parse_bench_type(s.T)),typemax(Int)÷(8maximum(x.gpus for x in members))))÷quantum)
        make(k) = candidate_runs(members,gs,raw,baseline_shape(s,k*quantum))
        # Evaluate once before search to surface unsupported model errors.
        all(r->peak_bytes(r.memory)<=budget,make(lo)) || error("Minimum problem does not fit in $(key[1])")
        best = largest_feasible(lo,hi,k->all(r->peak_bytes(r.memory)<=budget,make(k)))
        best === nothing && error("No feasible size for $(key[1])")
        append!(planned,make(best))
    end
    return planned
end

function print_plan(runs,budget)
    println("Per-GPU budget: $budget bytes; sizes are shared within each comparison group.")
    for r in runs
        m = r.memory
        println("$(r.spec.name) / $(r.backend) / $(r.spec.T) fusion=$(r.spec.fusion) GPUs=$(r.spec.gpus) N=$(r.N) M=$(r.M)")
        println("  initialization=$(m.initialization), iteration=$(m.iteration), workspace=$(m.workspace), peak=$(peak_bytes(m)) bytes")
        println("  $(m.explanation)")
    end
    limiting = runs[argmax([peak_bytes(r.memory) for r in runs])]
    println("Largest planned peak: $(limiting.spec.name) / $(limiting.backend) / $(limiting.spec.gpus) GPUs")
end
