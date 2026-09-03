using TOML

"""
One benchmark invocation parsed from `benchmarks.toml`. `name` selects the
benchmark type from `BENCHMARKS`; `T` is the element type (e.g. "Float32");
`args` are the sizes (`N M`) when pinned. When `autosize` is true, `args` is
unused and `N_hint` / `M_hint` feed `fit_one_gpu`.
"""
struct BenchmarkSpec
    name::String
    T::String
    gpus::Int
    cpus::Int
    fusion::Bool
    cuda::Bool
    n_warmup::Int
    n_iter::Int
    n_trial::Int
    args::Vector{Int}
    autosize::Bool
    N_hint::Union{Int,Nothing}
    M_hint::Union{Int,Nothing}
end

# A field may be a scalar or a list.
aslist(x) = x isa AbstractVector ? collect(x) : [x]

# `fusion` accepts a bool or "on"/"off" (or a list of these).
function parse_fusion(x)
    x isa Bool && return x
    s = lowercase(string(x))
    s in ("on", "true") && return true
    s in ("off", "false") && return false
    return error("fusion must be on/off (or true/false); got $(repr(x))")
end

# Value of a zipped field for sweep position `i`. length==1 field broadcasts.
sweep_value(field, i) = length(field) == 1 ? field[1] : field[i]

# Number of positions in the sweep. Every multi-element field must agree on length;
# length==1 fields broadcast and don't constrain it.
function sweep_length(name, fields)
    lengths = [length(field) for (_, field) in fields if length(field) > 1]
    isempty(lengths) && return 1
    allequal(lengths) || error(
        "benchmark '$(name)': zipped fields gpus/cpus/N/M must share one length " *
        "or be scalar; got " * join(("$k=$(length(v))" for (k, v) in fields), ", "),
    )
    return first(lengths)
end

# Names of the `[[name]]` blocks in the order they appear in the file. TOML.jl
# parses into an unordered Dict, so we scan the source to preserve run order.
function declared_order(path)
    order = String[]
    for line in eachline(path)
        header = strip(line)
        startswith(header, "[[") && endswith(header, "]]") || continue
        name = strip(header[3:(end - 2)])
        name in order || push!(order, name) # if not in list, push to ordered list
    end
    return order
end

function size_field(raw)
    raw === nothing && return (:omitted, Int[])
    vals = collect(aslist(raw))
    autos = [is_auto_size(v) for v in vals]
    if any(autos)
        all(autos) || error("cannot mix auto and numeric sizes in one field; got $(repr(raw))")
        return (:auto, Int[])
    end
    return (:pinned, Int[Int(v) for v in vals])
end

function parse_config(path)
    raw = TOML.parsefile(path)

    g = raw["Global"]
    global_settings = GlobalSettings(;
        n_warmup=g["n_warmup"], n_iter=g["n_iter"], n_trial=get(g, "n_trial", 1),
        cupynumeric=get(g, "cupynumeric", false),
        cuda=get(g, "cuda", false),
        check_correctness=get(g, "check_correctness", false),
        n_correctness_iter=get(g, "n_correctness_iter", 5),
        auto_size=get(g, "auto_size", false),
        mem_frac=Float64(get(g, "mem_frac", 0.5)),
    )

    specs = BenchmarkSpec[]
    for name in declared_order(path)
        entries = raw[name]
        entries isa AbstractVector || continue
        for e in entries
            types = aslist(get(e, "T", "Float32"))
            gpus = aslist(e["gpus"])
            cpus = aslist(e["cpus"])
            fusion = aslist(get(e, "fusion", true))
            nmode, nvals = size_field(get(e, "N", nothing))
            mmode, mvals = size_field(get(e, "M", nothing))
            cuda = get(e, "cuda", global_settings.cuda)
            n_warmup = get(e, "n_warmup", global_settings.n_warmup)
            n_iter = get(e, "n_iter", global_settings.n_iter)
            n_trial = get(e, "n_trial", global_settings.n_trial)
            block_auto = get(e, "auto_size", global_settings.auto_size)

            n_auto = nmode == :omitted || nmode == :auto
            m_auto = mmode == :auto
            use_auto = block_auto && (n_auto || m_auto)
            if use_auto
                nmode == :pinned && length(nvals) != 1 && error(
                    "benchmark '$(name)': autosize with pinned N requires a scalar N",
                )
                mmode == :pinned && length(mvals) != 1 && error(
                    "benchmark '$(name)': autosize with pinned M requires a scalar M",
                )
                N_hint = nmode == :pinned ? nvals[1] : nothing
                M_hint = mmode == :pinned ? mvals[1] : nothing
                n = sweep_length(name, ["gpus" => gpus, "cpus" => cpus])
            else
                nmode == :auto && error("benchmark '$(name)': N = \"auto\" requires auto_size = true")
                mmode == :auto && error("benchmark '$(name)': M = \"auto\" requires auto_size = true")
                nmode == :omitted && error(
                    "benchmark '$(name)' is missing N (set N or enable auto_size)",
                )
                mmode == :omitted && (mvals = [1])
                N_hint = nothing
                M_hint = nothing
                n = sweep_length(name, ["gpus" => gpus, "cpus" => cpus, "N" => nvals, "M" => mvals])
            end

            for T in types, fuse in fusion, i in 1:n
                args = use_auto ? Int[0, 0] : Int[sweep_value(nvals, i), sweep_value(mvals, i)]
                push!(
                    specs,
                    BenchmarkSpec(
                        name,
                        string(T),
                        Int(sweep_value(gpus, i)),
                        Int(sweep_value(cpus, i)),
                        parse_fusion(fuse),
                        cuda,
                        n_warmup,
                        n_iter,
                        n_trial,
                        args,
                        use_auto,
                        N_hint,
                        M_hint,
                    ),
                )
            end
        end
    end

    return global_settings, specs
end

# Group members that share a figure. Explicit `[plot.groups]` lists first;
# every other `[[benchmark]]` table is a singleton group named after itself.
# Group order follows the first listed member in the file.
function parse_plot_groups(path)
    raw = TOML.parsefile(path)
    declared = declared_order(path)
    plot = get(raw, "plot", Dict{String,Any}())
    groups_tbl = get(plot, "groups", Dict{String,Any}())

    explicit = Dict{String,Vector{String}}()
    for (gname, members) in groups_tbl
        explicit[string(gname)] = String[string(m) for m in aslist(members)]
    end

    assigned = Set{String}()
    groups = Pair{String,Vector{String}}[]
    remaining = copy(explicit)
    for name in declared
        name in assigned && continue
        gname = nothing
        for (g, members) in remaining
            if name in members
                gname = g
                break
            end
        end
        if gname !== nothing
            members = remaining[gname]
            push!(groups, gname => members)
            union!(assigned, members)
            delete!(remaining, gname)
        else
            push!(groups, name => [name])
            push!(assigned, name)
        end
    end
    for (gname, members) in remaining
        push!(groups, gname => members)
    end
    return groups
end

function plot_baseline(members::Vector{String})
    i = findfirst(m -> endswith(m, "_baseline"), members)
    i !== nothing && return members[i]
    return first(members)
end
