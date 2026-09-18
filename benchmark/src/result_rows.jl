struct Row
    gpus::Int
    N::Int
    M::Int
    time_ms::Float64
    thr::Float64
end

function load_runs(path)
    rows = Row[]
    for line in eachline(path)
        isempty(strip(line)) && continue
        f = split(line, ',')
        length(f) == 8 || error("Invalid result row in $path")
        push!(rows,Row(parse(Int,f[2]),parse(Int,f[3]),parse(Int,f[4]),parse(Float64,f[6]),parse(Float64,f[7])))
    end
    isempty(rows) && return Vector{Row}[]
    runs = [Row[]]
    for (i,r) in enumerate(rows)
        i > 1 && r.gpus < rows[i-1].gpus && push!(runs,Row[])
        push!(runs[end],r)
    end
    return runs
end

function aggregate(rows)
    by = Dict{Int,Vector{Row}}()
    for r in rows
        push!(get!(by,r.gpus,Row[]),r)
    end
    for (g,rs) in by
        length(unique((r.N,r.M) for r in rs)) == 1 || error("Cannot combine different dimensions at $g GPUs; select one invocation")
    end
    sd(x) = length(x)>1 ? std(x) : 0.0
    return [(gpus=g,N=first(by[g]).N,M=first(by[g]).M,
        t=mean(getfield.(by[g],:time_ms)),tsd=sd(getfield.(by[g],:time_ms)),
        h=mean(getfield.(by[g],:thr)),hsd=sd(getfield.(by[g],:thr))) for g in sort(collect(keys(by)))]
end

function validate_series_sizes(series)
    sizes = Dict{Int,Tuple{Int,Int}}()
    for s in series, r in s.agg
        previous = get!(sizes,r.gpus,(r.N,r.M))
        previous == (r.N,r.M) || error("Comparison series use different dimensions at $(r.gpus) GPUs")
    end
    return nothing
end
