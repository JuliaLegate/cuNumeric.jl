using TOML

"""
One benchmark invocation parsed from `benchmarks.toml`. `name` selects the
benchmark type from `BENCHMARKS`; `T` is the element type (e.g. "Float32");
`args` are the sizes (currently `N M`).
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

function parse_config(path)
    raw = TOML.parsefile(path)

    g = raw["Global"]
    global_settings = GlobalSettings(;
        n_warmup=g["n_warmup"], n_iter=g["n_iter"], n_trial=get(g, "n_trial", 1),
        cupynumeric=get(g, "cupynumeric", false),
        cuda=get(g, "cuda", false),
        check_correctness=get(g, "check_correctness", false),
        n_correctness_iter=get(g, "n_correctness_iter", 5),
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
            N = aslist(e["N"])
            M = aslist(get(e, "M", 1))
            cuda = get(e, "cuda", global_settings.cuda)
            n_warmup = get(e, "n_warmup", global_settings.n_warmup)
            n_iter = get(e, "n_iter", global_settings.n_iter)
            n_trial = get(e, "n_trial", global_settings.n_trial)

            n = sweep_length(name, ["gpus" => gpus, "cpus" => cpus, "N" => N, "M" => M])

            for T in types, fuse in fusion, i in 1:n
                push!(
                    specs,
                    BenchmarkSpec(
                        name,
                        T,
                        sweep_value(gpus, i),
                        sweep_value(cpus, i),
                        parse_fusion(fuse),
                        cuda,
                        n_warmup,
                        n_iter,
                        n_trial,
                        [sweep_value(N, i), sweep_value(M, i)],
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
