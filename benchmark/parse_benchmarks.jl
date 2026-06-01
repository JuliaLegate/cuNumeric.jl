using TOML

Base.@kwdef struct GlobalSettings
    N_warmup::Int
    N_iter::Int
    N_GPU::Int
end

function to_symbol_dict(d)
    return Dict(Symbol(k) => v for (k, v) in d)
end

function parse_config(path)
    raw = TOML.parsefile(path)

    global_settings = GlobalSettings(; to_symbol_dict(raw["Global"])...)

    benchmarks = AbstractBenchmark[]

    for (name, entries) in raw
        name == "Global" && continue

        BenchmarkType = getproperty(Main, Symbol(name))

        for entry in entries
            T = getproperty(Main, Symbol(entry["T"]))

            params = Dict{Symbol,Any}()
            for (k, v) in entry
                k == "T" && continue
                params[Symbol(k)] = v
            end

            push!(benchmarks, BenchmarkType{T}(; params...))
        end
    end

    return global_settings, benchmarks
end
