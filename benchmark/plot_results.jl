#!/usr/bin/env julia
# Generate weak-scaling plots from benchmark CSVs.
# Each figure shows throughput, time per step, and parallel efficiency.

using Plots
using Statistics

gr()

function parse_args(args)
    results_dir = "results"
    out_dir = nothing
    output_suffix = ""

    for arg in args
        if startswith(arg, "--out=")
            out_dir = last(split(arg, "="; limit=2))
        elseif startswith(arg, "--suffix=")
            output_suffix = last(split(arg, "="; limit=2))
        else
            results_dir = arg
        end
    end
    results_dir = isabspath(results_dir) ? results_dir : joinpath(@__DIR__, results_dir)
    if out_dir === nothing
        out_dir = if basename(normpath(results_dir)) == "results"
            joinpath(@__DIR__, "plots")
        else
            joinpath(@__DIR__, "plots", basename(normpath(results_dir)))
        end
    else
        out_dir = isabspath(out_dir) ? out_dir : joinpath(@__DIR__, out_dir)
    end
    return (; results_dir, out_dir, output_suffix)
end

const CONFIG = parse_args(ARGS)
const RESULTS_DIR = CONFIG.results_dir
const OUT_DIR = CONFIG.out_dir
const OUTPUT_SUFFIX = CONFIG.output_suffix

# file key, display label, color, marker
const FAMILIES = [
    ("cunumeric", "cuNumeric.jl (fused)", "#2a78d6", :circle),
    ("cunumeric_nofusion", "cuNumeric.jl (unfused)", "#4a3aa7", :diamond),
    ("cupynumeric", "cuPyNumeric", "#eb6834", :rect),
    ("CUDA.jl", "CUDA.jl", "#008300", :utriangle),
]

const INK = "#0b0b0b"
const MUTED = "#898781"
const GRIDCOL = "#e1e0d9"
const IDEALCOL = "#c3c2b7"
const GROUP_COLORS = ["#2a78d6", "#eb6834", "#008300", "#8b3fb0", "#c47f00", "#159a9c"]
const GROUP_MARKERS = [:circle, :diamond, :utriangle, :rect, :star5, :hexagon]

struct Row
    gpus::Int
    time_ms::Float64
    thr::Float64
end

# Parse a CSV into runs, split wherever the GPU count resets to a smaller value.
function load_runs(path)
    rows = Row[]
    for line in eachline(path)
        isempty(strip(line)) && continue
        f = split(line, ',')
        push!(rows, Row(parse(Int, f[2]), parse(Float64, f[6]), parse(Float64, f[7])))
    end
    isempty(rows) && return Vector{Row}[]
    runs = [Row[]]
    for (i, r) in enumerate(rows)
        i > 1 && r.gpus < rows[i - 1].gpus && push!(runs, Row[])
        push!(runs[end], r)
    end
    return runs
end

# Aggregate trials per GPU count -> sorted vector of (gpus, t, tsd, h, hsd).
function aggregate(rows)
    by = Dict{Int,Vector{Row}}()
    for r in rows
        push!(get!(by, r.gpus, Row[]), r)
    end
    sd(x) = length(x) > 1 ? std(x) : 0.0
    return [
        (gpus=g, t=mean(getfield.(by[g], :time_ms)), tsd=sd(getfield.(by[g], :time_ms)),
            h=mean(getfield.(by[g], :thr)), hsd=sd(getfield.(by[g], :thr)))
        for g in sort(collect(keys(by)))
    ]
end

_all_rows(runs) = reduce(vcat, runs; init=Row[])

function make_series(label, color, marker, ls, runs)
    isempty(runs) && return nothing
    return (label=label, color=color, marker=marker, ls=ls,
        agg=aggregate(_all_rows(runs)))
end

# Build the available implementation series for one benchmark.
function load_series(bench, specs; filename=(b, k) -> "$(b)_$(k).csv")
    series = []
    for (key, label, color, marker) in specs
        path = joinpath(RESULTS_DIR, filename(bench, key))
        isfile(path) || continue
        runs = load_runs(path)
        isempty(runs) && continue

        push!(series, make_series(label, color, marker, :solid, runs))
    end
    return filter(!isnothing, series)
end

series_for(bench) = load_series(bench, FAMILIES)

function common_prefix(names)
    words = split.(names, '_')
    n = minimum(length, words)
    i = 0
    while i < n && all(w -> w[i + 1] == words[1][i + 1], words)
        i += 1
    end
    return i == 0 ? String[] : words[1][1:i]
end

function family_stem(name, prefix)
    words = split(name, '_')
    length(words) > length(prefix) || return name
    return join(words[(length(prefix) + 1):end], "_")
end

function grouped_series(benches, key)
    series = []
    prefix = common_prefix(benches)
    for (i, bench) in enumerate(benches)
        path = joinpath(RESULTS_DIR, "$(bench)_$(key).csv")
        isfile(path) || continue
        runs = load_runs(path)
        isempty(runs) && continue
        color = GROUP_COLORS[mod1(i, length(GROUP_COLORS))]
        marker = GROUP_MARKERS[mod1(i, length(GROUP_MARKERS))]
        words = split(bench, '_')
        length(words) > length(prefix) && (words = words[(length(prefix) + 1):end])
        label = titlecase(join(words, ' '))
        push!(series, make_series(label, color, marker, :solid, runs))
    end
    return filter(!isnothing, series)
end

function throughput_label()
    return "Throughput"
end

function addline!(p, s, y; kw...)
    return plot!(p, getfield.(s.agg, :gpus), y; color=s.color,
        lw=2.2, ls=s.ls, marker=s.marker, ms=6, msc=s.color, markerstrokewidth=0.8,
        label=s.label, kw...)
end

# One legend key: a short line sample (+ optional marker) with a text label.
function swatch!(p, x, y, color, ls, marker, label)
    plot!(p, [x, x + 0.032], [y, y]; color=color, lw=2.6, ls=ls, label="")
    marker !== nothing && scatter!(p, [x + 0.016], [y]; color=color, marker=marker,
        ms=6, msc=color, markerstrokewidth=0.8, label="")
    return annotate!(p, x + 0.045, y, text(label, 9, INK, :left))
end

# Build a legend from the series currently being plotted.
function build_legend(series)
    pl = plot(; framestyle=:none, legend=false, xlims=(0, 1), ylims=(0, 1),
        grid=false, ticks=false)
    present = [(s.label, s.color, s.marker) for s in series]
    annotate!(pl, 0.015, 0.74, text("Series", 10, INK, :left))
    xs = range(0.18, 0.80; length=max(length(present), 1))
    for ((fam, color, marker), x) in zip(present, xs)
        swatch!(pl, x, 0.74, color, :solid, marker, fam)
    end
    swatch!(pl, 0.18, 0.26, IDEALCOL, :dashdot, nothing, "efficiency = 1")
    return pl
end

function positive_ylim(vals; pad=0.12)
    isempty(vals) && return (0, 1)
    hi = maximum(vals)
    hi > 0 || return (0, 1)
    return (0, hi * (1 + pad))
end

function weak_scaling_figure(bench, series, legend_panel; plot_title)
    common = (xscale=:log2, xticks=([1, 2, 4, 8], ["1", "2", "4", "8"]), xlabel="GPUs",
        framestyle=:box, grid=true, gridcolor=GRIDCOL, gridalpha=1.0,
        foreground_color_text=INK, tickfontcolor=MUTED, legend=false,
        xlims=(0.85, 9.4))

    # Panel 1: throughput (higher better)
    throughput = [x.h for s in series for x in s.agg]
    p1 = plot(; ylabel=throughput_label(), title="Throughput",
        ylims=positive_ylim(throughput), common...)
    for s in series
        addline!(p1, s, getfield.(s.agg, :h); yerror=getfield.(s.agg, :hsd))
    end

    # Panel 2: time per step (lower better; ideal = flat)
    p2 = plot(; ylabel="Time / step (ms)", title="Time per step", common...)
    for s in series
        addline!(p2, s, getfield.(s.agg, :t); yerror=getfield.(s.agg, :tsd))
    end

    # Panel 3: parallel efficiency = thr(p)/(p*thr(1)); ideal = 1.0
    efficiencies = Float64[]
    for s in series
        i1 = findfirst(x -> x.gpus == 1, s.agg)
        i1 === nothing && continue
        base = s.agg[i1].h
        append!(efficiencies, [x.h/(x.gpus*base) for x in s.agg])
    end
    p3 = plot(; ylabel="Parallel efficiency", title="Weak-scaling efficiency",
        ylims=positive_ylim(vcat(efficiencies, [1.0])), common...)
    hline!(p3, [1.0]; color=IDEALCOL, ls=:dashdot, lw=1.4, label="")
    for s in series
        i1 = findfirst(x -> x.gpus == 1, s.agg)
        i1 === nothing && continue
        base = s.agg[i1].h
        addline!(p3, s, [x.h/(x.gpus*base) for x in s.agg])
    end

    return plot(
        p1, p2, p3, legend_panel;
        layout=@layout([grid(1, 3); leg{0.16h}]),
        size=(1400, 600), dpi=200, plot_title,
        plot_titlefontsize=12, left_margin=6Plots.mm,
        bottom_margin=6Plots.mm, top_margin=4Plots.mm,
        background_color="#fcfcfb",
    )
end

function main()
    mkpath(OUT_DIR)
    files = filter(f -> endswith(f, ".csv"), readdir(RESULTS_DIR))
    benches = unique(
        String[
            m.captures[1] for f in files for (key, _, _, _) in FAMILIES
            for m in (match(Regex("^(.*)_" * replace(key, "." => "\\.") * "\\.csv\$"), f),)
            if m !== nothing
        ],
    )
    family = common_prefix(benches)
    benchmark_out = joinpath(OUT_DIR, isempty(family) ? "benchmarks" : join(family, "_"))
    mkpath(benchmark_out)

    for bench in benches
        series = series_for(bench)
        isempty(series) && continue

        fig = weak_scaling_figure(
            bench, series, build_legend(series);
            plot_title=titlecase(bench) * " — weak scaling",
        )

        stem = family_stem(bench, family)
        out = joinpath(benchmark_out, "$(stem)_weak_scaling$(OUTPUT_SUFFIX).png")
        savefig(fig, out)
        println("wrote $out")
    end

    # Add aggregate views that compare all benchmark variants for each mode.
    for (key, title, stem) in (("cunumeric", "Fusion enabled", "fusion"),
        ("cunumeric_nofusion", "Fusion disabled", "no_fusion"))
        series = grouped_series(benches, key)
        isempty(series) && continue
        fig = weak_scaling_figure(
            stem, series, build_legend(series);
            plot_title=title * " — weak scaling",
        )
        out = joinpath(benchmark_out, "$(stem)_weak_scaling$(OUTPUT_SUFFIX).png")
        savefig(fig, out)
        println("wrote $out")
    end
    return nothing
end

main()
