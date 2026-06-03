using TOML

"""
One benchmark invocation parsed from `benchmarks.toml`. `name` selects the
benchmark type from `BENCHMARKS`; `T` is the element type (e.g. "Float32");
`variant` names the run variant (e.g. "baseline", "lifetimes"); `args` are the
sizes (currently `N M`).
"""
struct BenchmarkSpec
    name::String
    T::String
    variant::String
    gpus::Int
    cpus::Int
    args::Vector{Int}
end

# A field may be a scalar or a list.
aslist(x) = x isa AbstractVector ? collect(x) : [x]

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

function parse_config(path)
    raw = TOML.parsefile(path)

    g = raw["Global"]
    global_settings = GlobalSettings(;
        n_warmup=g["n_warmup"], n_iter=g["n_iter"], n_trial=get(g, "n_trial", 1)
    )

    specs = BenchmarkSpec[]
    for (name, entries) in raw
        name == "Global" && continue
        for e in entries
            types = aslist(get(e, "T", "Float32"))
            variants = aslist(get(e, "variants", "baseline"))
            gpus = aslist(e["gpus"])
            cpus = aslist(e["cpus"])
            N = aslist(e["N"])
            M = aslist(get(e, "M", 1))

            n = sweep_length(name, ["gpus" => gpus, "cpus" => cpus, "N" => N, "M" => M])

            # `T` and `variants` multiply; gpus/cpus/N/M zip into the sweep.
            for T in types, variant in variants, i in 1:n
                push!(
                    specs,
                    BenchmarkSpec(
                        name, T, variant, sweep_value(gpus, i), sweep_value(cpus, i),
                        [sweep_value(N, i), sweep_value(M, i)],
                    ),
                )
            end
        end
    end

    return global_settings, specs
end
