#!/usr/bin/env julia
# Weak-scaling plots (1/2/4/8 GPUs) for the benchmark result CSVs.
# One figure per benchmark, three panels: throughput, time/step, parallel efficiency.
#
# CSV schema (see src/core.jl save_result):
# implementation,gpus,N,M,trial,time_ms,throughput,correctness
# `throughput` is the benchmark's `total_flops` divided by elapsed time. For
# Gray-Scott that unit is Gpoint-updates/s; other benchmarks report GFLOP/s.
#
# save_result appends and does NOT encode the code-path variant, so a cuNumeric CSV
# holds alternating runs: baseline, @accelerate, baseline, @accelerate, ...
# (a run boundary = the GPU count resetting downward).
# cuPyNumeric / CUDA.jl have no accelerated path -> a single block.
#
# Encoding: color = implementation; line style = code path
#   solid = baseline, dashed = @accelerate.

using Plots
using Statistics

gr()

function parse_args(args)
    results_dir = "results"
    out_dir = nothing
    single_cunumeric_run = :baseline
    hide_baseline = false
    output_suffix = ""

    for arg in args
        if startswith(arg, "--single-cu=")
            value = Symbol(lowercase(last(split(arg, "="; limit=2))))
            value in (:baseline, :accelerated) ||
                error("--single-cu must be baseline or accelerated")
            single_cunumeric_run = value
        elseif arg == "--hide-baseline"
            hide_baseline = true
        elseif startswith(arg, "--out=")
            out_dir = last(split(arg, "="; limit=2))
        elseif startswith(arg, "--suffix=")
            output_suffix = last(split(arg, "="; limit=2))
        else
            results_dir = arg
        end
    end
    isempty(output_suffix) && hide_baseline && (output_suffix = "_no_baseline")

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
    return (; results_dir, out_dir, single_cunumeric_run, hide_baseline, output_suffix)
end

const CONFIG = parse_args(ARGS)
const RESULTS_DIR = CONFIG.results_dir
const OUT_DIR = CONFIG.out_dir
const SINGLE_CUNUMERIC_RUN = CONFIG.single_cunumeric_run
const HIDE_BASELINE = CONFIG.hide_baseline
const OUTPUT_SUFFIX = CONFIG.output_suffix

# filekey, family label, color, marker, can_contain_accelerated_blocks
const FAMILIES = [
    ("cunumeric", "cuNumeric.jl (fused)", "#2a78d6", :circle, true),
    ("cunumeric_nofusion", "cuNumeric.jl (unfused)", "#4a3aa7", :diamond, true),
    ("cupynumeric", "cuPyNumeric", "#eb6834", :rect, false),
    ("CUDA.jl", "CUDA.jl", "#008300", :utriangle, false),
]

const INK = "#0b0b0b"
const MUTED = "#898781"
const GRIDCOL = "#e1e0d9"
const IDEALCOL = "#c3c2b7"

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

# Build the series (color+marker+linestyle+agg) present for one benchmark.
function series_for(bench)
    series = []  # NamedTuple(label,color,marker,ls,agg)
    for (key, fam, color, marker, splits) in FAMILIES
        path = joinpath(RESULTS_DIR, "$(bench)_$(key).csv")
        isfile(path) || continue
        runs = load_runs(path)
        isempty(runs) && continue
        if splits && bench == "grayscott"
            if length(runs) == 1
                label =
                    SINGLE_CUNUMERIC_RUN === :accelerated ? "$fam · accelerated" :
                    "$fam · baseline"
                ls = SINGLE_CUNUMERIC_RUN === :accelerated ? :dash : :solid
                push!(
                    series, (label=label, color=color, marker=marker, ls=ls,
                        agg=aggregate(runs[1]))
                )
            else
                # Repeated harness runs append alternating baseline/accelerated blocks.
                baseline_rows = reduce(vcat, runs[1:2:end])
                accelerated_rows = reduce(vcat, runs[2:2:end])
                push!(
                    series,
                    (label="$fam · baseline", color=color, marker=marker,
                        ls=:solid, agg=aggregate(baseline_rows)),
                )
                push!(series,
                    (label="$fam · accelerated", color=color, marker=marker,
                        ls=:dash, agg=aggregate(accelerated_rows)))
            end
        else
            push!(
                series,
                (label=fam, color=color, marker=marker, ls=:solid,
                    agg=aggregate(reduce(vcat, runs))),
            )
        end
    end
    return series
end

function throughput_label(bench)
    return bench == "grayscott" ?
           "Throughput (Gpoint-updates/s)" : "Throughput (GFLOP/s)"
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

# Grouped legend: color/marker = implementation, line style = code path.
function build_legend(series)
    pl = plot(; framestyle=:none, legend=false, xlims=(0, 1), ylims=(0, 1),
        grid=false, ticks=false)
    # implementations present, in FAMILIES order, matched by color
    present = [
        (fam, color, marker) for (key, fam, color, marker, _) in FAMILIES
        if any(s.color == color for s in series)
    ]
    annotate!(pl, 0.015, 0.74, text("Implementation", 10, INK, :left))
    xs = range(0.18, 0.80; length=max(length(present), 1))
    for ((fam, color, marker), x) in zip(present, xs)
        swatch!(pl, x, 0.74, color, :solid, marker, fam)
    end
    annotate!(pl, 0.015, 0.26, text("Line style", 10, INK, :left))
    has_baseline = any(endswith(s.label, "· baseline") for s in series)
    has_accelerated = any(endswith(s.label, "· accelerated") for s in series)
    if has_baseline && has_accelerated
        swatch!(pl, 0.18, 0.26, MUTED, :solid, nothing, "baseline")
        swatch!(pl, 0.40, 0.26, MUTED, :dash, nothing, "@accelerate")
        swatch!(pl, 0.70, 0.26, IDEALCOL, :dashdot, nothing, "efficiency = 1")
    elseif has_accelerated
        swatch!(pl, 0.18, 0.26, MUTED, :dash, nothing, "@accelerate")
        swatch!(pl, 0.52, 0.26, IDEALCOL, :dashdot, nothing, "efficiency = 1")
    elseif has_baseline
        swatch!(pl, 0.18, 0.26, MUTED, :solid, nothing, "baseline")
        swatch!(pl, 0.48, 0.26, IDEALCOL, :dashdot, nothing, "efficiency = 1")
    else
        swatch!(pl, 0.18, 0.26, IDEALCOL, :dashdot, nothing, "efficiency = 1")
    end
    return pl
end

function positive_ylim(vals; pad=0.12)
    isempty(vals) && return (0, 1)
    hi = maximum(vals)
    hi > 0 || return (0, 1)
    return (0, hi * (1 + pad))
end

function main()
    mkpath(OUT_DIR)
    files = filter(f -> endswith(f, ".csv"), readdir(RESULTS_DIR))
    benches = unique(
        String[
            m.captures[1] for f in files for (key, _, _, _, _) in FAMILIES
            for m in (match(Regex("^(.*)_" * replace(key, "." => "\\.") * "\\.csv\$"), f),)
            if m !== nothing
        ],
    )

    for bench in benches
        series = series_for(bench)
        if HIDE_BASELINE
            series = filter(s -> !endswith(s.label, "· baseline"), series)
        end
        isempty(series) && continue

        common = (xscale=:log2, xticks=([1, 2, 4, 8], ["1", "2", "4", "8"]), xlabel="GPUs",
            framestyle=:box, grid=true, gridcolor=GRIDCOL, gridalpha=1.0,
            foreground_color_text=INK, tickfontcolor=MUTED, legend=false,
            xlims=(0.85, 9.4))

        # Panel 1: throughput (higher better)
        throughput = [x.h for s in series for x in s.agg]
        p1 = plot(; ylabel=throughput_label(bench), title="Throughput",
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

        # grouped legend panel: color/marker = implementation, style = code path
        pl = build_legend(series)
        has_baseline = any(endswith(s.label, "· baseline") for s in series)
        has_accelerated = any(endswith(s.label, "· accelerated") for s in series)
        style_title = if has_baseline && has_accelerated
            "solid = baseline · dashed = @accelerate"
        elseif has_accelerated
            "dashed = @accelerate"
        elseif has_baseline
            "solid = baseline"
        else
            "implementation comparison"
        end

        fig = plot(p1, p2, p3, pl; layout=@layout([grid(1, 3); leg{0.16h}]),
            size=(1400, 600), dpi=200,
            plot_title=titlecase(bench) * " — weak scaling  ($style_title)",
            plot_titlefontsize=12, left_margin=6Plots.mm,
            bottom_margin=6Plots.mm, top_margin=4Plots.mm,
            background_color="#fcfcfb")

        out = joinpath(OUT_DIR, "$(bench)_weak_scaling$(OUTPUT_SUFFIX).png")
        savefig(fig, out)
        println("wrote $out")
    end
end

main()
