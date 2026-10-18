# Usage: julia --project=ENV scripts/latency/measure_accelerate.jl [samples] [modes...]
# Dependencies must already be instantiated and precompiled. This driver does
# not load cuNumeric; every measurement runs in a separate Julia process.
using Statistics: median

samples = isempty(ARGS) ? 3 : parse(Int, first(ARGS))
samples > 0 || error("samples must be positive")
modes = length(ARGS) > 1 ? ARGS[2:end] :
    ["precompile", "function", "begin", "multi", "let", "expression", "stencil", "plain"]
worker = joinpath(@__DIR__, "accelerate_worker.jl")
project = dirname(Base.active_project())
measurements = Dict{Tuple{String,String},Vector{Float64}}()

for mode in modes, sample in 1:samples
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$project $worker $mode`
    # Package compilation must work without a running Legate runtime. Ordinary
    # load/operation samples honor the caller's runtime configuration.
    if mode == "precompile"
        cmd = addenv(cmd, "LEGATE_SKIP_RUNTIME" => "true")
    end
    output = read(cmd, String)
    println("SAMPLE,", mode, ",", sample)
    print(output)
    flush(stdout)
    for line in split(output, '\n')
        parts = split(line, ',')
        length(parts) == 3 && first(parts) == "LATENCY" || continue
        values = get!(Vector{Float64}, measurements, (mode, parts[2]))
        push!(values, parse(Float64, parts[3]))
    end
end

println("mode,metric,median_seconds,min_seconds,max_seconds,samples")
for ((mode, metric), values) in sort!(collect(measurements); by=first)
    println(join(
        (mode, metric, median(values), minimum(values), maximum(values), length(values)), ','
    ))
end
