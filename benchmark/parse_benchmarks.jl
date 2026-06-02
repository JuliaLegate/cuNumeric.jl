using TOML

function to_symbol_dict(d)
    return Dict(Symbol(k) => v for (k, v) in d)
end

function parse_config(path)
    raw = TOML.parsefile(path)

    global_settings = GlobalSettings(; to_symbol_dict(raw["Global"])...)

    benchmarks = AbstractBenchmark[]

    for (name, entries) in raw
        name == "Global" && continue

        # Convert name parsed as String, to actual type
        BenchmarkType = getproperty(Main, Symbol(name))

        for entry in entries
            # Convert type parsed as String, to actual type
            T = getproperty(Main, Symbol(entry["T"]))

            params = Dict{Symbol,Any}()
            for (k, v) in entry
                k == "T" && continue
                params[Symbol(k)] = v
            end

            if T <: allowed_types(BenchmarkType)
                push!(benchmarks, BenchmarkType{T}(; params...))
            else
                @warn "$(BenchmarkType) does not support benchmarking with type $(T). Skipping."
            end
        end
    end

    return global_settings, benchmarks
end
