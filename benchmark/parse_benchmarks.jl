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
    args::Vector{Int}
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
            push!(
                specs,
                BenchmarkSpec(name, get(e, "T", "Float32"), e["gpus"], e["cpus"], [e["N"], e["M"]]),
            )
        end
    end

    return global_settings, specs
end
