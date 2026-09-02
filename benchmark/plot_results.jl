#!/usr/bin/env julia
# Generate weak-scaling plots from benchmark CSVs.
# Each figure is one `[plot.groups]` entry (or a singleton [[benchmark]]).

using Plots
using Statistics

include(joinpath(@__DIR__, "src", "parse_benchmarks.jl"))

# GPU nodes often have no display; 100 = PNG. Override with GKSwstype if needed.
get!(ENV, "GKSwstype", "100")
gr()
default(;
    fontfamily="sans-serif",
    fg=:black,
    fg_text=:black,
    fg_axis=:black,
    fg_border=:black,
    grid=false,
    gridalpha=0,
    gridlinewidth=0,
    minorgrid=false,
    legend=false,
)

function parse_args(args)
    results_dir = "results"
    out_dir = nothing
    output_suffix = ""
    config = joinpath(@__DIR__, "benchmarks.toml")

    for arg in args
        if startswith(arg, "--out=")
            out_dir = last(split(arg, "="; limit=2))
        elseif startswith(arg, "--suffix=")
            output_suffix = last(split(arg, "="; limit=2))
        elseif startswith(arg, "--config=")
            config = last(split(arg, "="; limit=2))
        else
            results_dir = arg
        end
    end
    results_dir = isabspath(results_dir) ? results_dir : joinpath(@__DIR__, results_dir)
    config = isabspath(config) ? config : joinpath(@__DIR__, config)
    if out_dir === nothing
        out_dir = if basename(normpath(results_dir)) == "results"
            joinpath(@__DIR__, "plots")
        else
            joinpath(@__DIR__, "plots", basename(normpath(results_dir)))
        end
    else
        out_dir = isabspath(out_dir) ? out_dir : joinpath(@__DIR__, out_dir)
    end
    return (; results_dir, out_dir, output_suffix, config)
end

# Fixed across every figure so GEMM / Gray-Scott / DMD read as one set.
const COLOR_CUNUMERIC = "#2a78d6"
const COLOR_CUPYNUMERIC = "#eb6834"
const COLOR_CUDA = "#1a7f37"
const COLOR_CUTENSOR = "#0d7377"
const MARKER_CUNUMERIC = :circle
const MARKER_CUPYNUMERIC = :rect
const MARKER_CUDA = :utriangle
const MARKER_CUTENSOR = :star5

# Extra cuNumeric variants (Gray-Scott forms, DMD accelerated). Avoid the
# reference orange/green so CUDA.jl and cuPyNumeric stay unique.
const VARIANT_COLORS = [COLOR_CUNUMERIC, "#7b2d8e", "#b8860b", "#3d5a80", "#a23b72", "#2f6f4e"]
const VARIANT_MARKERS = [:circle, :diamond, :hexagon, :dtriangle, :star4, :pentagon]

const REF_FAMILIES = [
    ("cupynumeric", "cuPyNumeric", COLOR_CUPYNUMERIC, MARKER_CUPYNUMERIC),
    ("CUDA.jl", "CUDA.jl", COLOR_CUDA, MARKER_CUDA),
    ("tensoroperations_cuda", "TensorOperations.jl / cuTENSOR", COLOR_CUTENSOR, MARKER_CUTENSOR),
]

const INK = "#111111"
const IDEALCOL = "#6e6e6e"

const GROUP_TITLES = Dict(
    "grayscott" => "Gray-Scott",
    "dmd" => "DMD",
    "gemm" => "GEMM",
    "poisson_fft" => "Poisson FFT",
    "montecarlo" => "Monte Carlo",
    "tensor_projection3" => "Tensor projection (3-mode)",
    "tensor_contract4" => "Tensor contraction (rank-4)",
)

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

function load_csv_series(results_dir, bench, key, label, color, marker, ls)
    path = joinpath(results_dir, "$(bench)_$(key).csv")
    isfile(path) || return nothing
    runs = load_runs(path)
    isempty(runs) && return nothing
    return make_series(label, color, marker, ls, runs)
end

function group_title(group)
    return get(GROUP_TITLES, group, titlecase(replace(group, '_' => ' ')))
end

function variant_label(group, member)
    member == group && return "cuNumeric.jl"
    prefix = group * "_"
    stem = startswith(member, prefix) ? member[(length(prefix) + 1):end] : member
    return replace(stem, '_' => ' ')
end

function cunumeric_series_label(group, member, fused)
    base = variant_label(group, member)
    return fused ? base : "$(base) (unfused)"
end

function overlay_refs(results_dir, members)
    series = []
    seen = Set{String}()
    order = unique!(vcat([plot_baseline(members)], members))
    for (key, label, color, marker) in REF_FAMILIES
        key in seen && continue
        for member in order
            s = load_csv_series(results_dir, member, key, label, color, marker, :solid)
            s === nothing && continue
            push!(series, s)
            push!(seen, key)
            break
        end
    end
    return series
end

function group_series(results_dir, group, members)
    series = []
    n_members = length(members)
    for (i, member) in enumerate(members)
        color, marker = if n_members == 1
            COLOR_CUNUMERIC, MARKER_CUNUMERIC
        else
            VARIANT_COLORS[mod1(i, length(VARIANT_COLORS))],
            VARIANT_MARKERS[mod1(i, length(VARIANT_MARKERS))]
        end
        for (key, fused, ls) in (
            ("cunumeric", true, :solid),
            ("cunumeric_nofusion", false, :dash),
        )
            label = cunumeric_series_label(group, member, fused)
            s = load_csv_series(results_dir, member, key, label, color, marker, ls)
            s === nothing || push!(series, s)
        end
    end
    append!(series, overlay_refs(results_dir, members))
    return filter(!isnothing, series)
end

function addline!(p, s, y; kw...)
    return plot!(p, getfield.(s.agg, :gpus), y; color=s.color,
        lw=2.6, ls=s.ls, marker=s.marker, ms=7, msc=s.color, markerstrokewidth=0.7,
        label=s.label, kw...)
end

function build_legend(series)
    n = length(series)
    cols = min(max(n, 1), 4)
    rows = cld(n, cols)
    slot = min(0.32, 0.92 / cols)
    x0 = (1 - cols * slot) / 2
    y0 = 0.50 + 0.18 * (rows - 1) / 2

    pl = plot(;
        framestyle=:none, grid=false, ticks=false, legend=false,
        xlims=(0, 1), ylims=(0, 1), widen=false,
        left_margin=0Plots.mm, right_margin=0Plots.mm,
        top_margin=0Plots.mm, bottom_margin=0Plots.mm,
        background_color=:transparent,
    )
    # Pin the coordinate system so later scatter/annotate cannot rescale it.
    scatter!(pl, [0.0, 1.0], [0.0, 1.0]; ms=0, msw=0, mc=:white, label="")

    for (i, s) in enumerate(series)
        r, c = divrem(i - 1, cols)
        x = x0 + c * slot
        y = y0 - r * 0.36
        plot!(pl, [x, x + 0.028], [y, y]; color=s.color, lw=2.8, ls=s.ls, label="")
        scatter!(pl, [x + 0.014], [y]; color=s.color, marker=s.marker,
            ms=7, msc=s.color, markerstrokewidth=0.6, label="")
        annotate!(pl, x + 0.036, y, text(s.label, 12, :black, :left))
    end
    plot!(pl; xlims=(0, 1), ylims=(0, 1), widen=false)
    return pl
end

function series_ymax(series, yfield, efield)
    m = 0.0
    for s in series, x in s.agg
        m = max(m, getfield(x, yfield) + getfield(x, efield))
    end
    return m
end

function positive_ylim(hi; pad=0.18)
    hi > 0 || return (0, 1)
    return (0, hi * (1 + pad))
end

function weak_scaling_figure(series; plot_title)
    common = (
        xscale=:log2, xticks=([1, 2, 4, 8], ["1", "2", "4", "8"]), xlabel="GPUs",
        framestyle=:box, grid=false, gridalpha=0, gridlinewidth=0, minorgrid=false,
        foreground_color_grid=:white,
        foreground_color_axis=:black, foreground_color_border=:black,
        foreground_color_text=:black, foreground_color_guide=:black,
        tickfontcolor=:black, guidefontcolor=:black, titlefontcolor=:black,
        tickfontsize=14, guidefontsize=16, titlefontsize=16,
        xtickfontsize=14, ytickfontsize=14,
        xguidefontsize=16, yguidefontsize=16,
        legend=false, xlims=(0.85, 9.4), widen=false,
        left_margin=10Plots.mm, right_margin=6Plots.mm,
        top_margin=5Plots.mm, bottom_margin=10Plots.mm,
    )

    p1 = plot(; ylabel="Throughput", title="Throughput",
        ylims=positive_ylim(series_ymax(series, :h, :hsd); pad=0.28), common...)
    for s in series
        addline!(p1, s, getfield.(s.agg, :h); yerror=getfield.(s.agg, :hsd))
    end

    p2 = plot(; ylabel="Time / step (ms)", title="Time per step",
        ylims=positive_ylim(series_ymax(series, :t, :tsd); pad=0.28),
        common..., left_margin=28Plots.mm, yguidefontsize=15)
    for s in series
        addline!(p2, s, getfield.(s.agg, :t); yerror=getfield.(s.agg, :tsd))
    end

    efficiencies = Float64[]
    for s in series
        i1 = findfirst(x -> x.gpus == 1, s.agg)
        i1 === nothing && continue
        base = s.agg[i1].h
        append!(efficiencies, [x.h / (x.gpus * base) for x in s.agg])
    end
    p3 = plot(; ylabel="Parallel efficiency", title="Weak-scaling efficiency",
        ylims=positive_ylim(max(1.0, isempty(efficiencies) ? 0.0 : maximum(efficiencies)); pad=0.12),
        common..., left_margin=16Plots.mm)
    hline!(p3, [1.0]; color=IDEALCOL, ls=:dashdot, lw=1.6, label="")
    for s in series
        i1 = findfirst(x -> x.gpus == 1, s.agg)
        i1 === nothing && continue
        base = s.agg[i1].h
        addline!(p3, s, [x.h / (x.gpus * base) for x in s.agg])
    end

    nrows = cld(length(series), 4)
    layout = if nrows > 1
        @layout([grid(1, 3); leg{0.16h}])
    else
        @layout([grid(1, 3); leg{0.14h}])
    end
    return plot(
        p1, p2, p3, build_legend(series);
        layout,
        size=(1760, nrows > 1 ? 680 : 620), dpi=220, plot_title,
        plot_titlefontsize=20, plot_titlefontcolor=:black,
        background_color=:white,
    )
end

function main(args=ARGS)
    cfg = parse_args(args)
    if !isdir(cfg.results_dir)
        println("no results directory at $(cfg.results_dir)")
        return nothing
    end
    isfile(cfg.config) || error("plot config not found: $(cfg.config)")

    mkpath(cfg.out_dir)
    for (group, members) in parse_plot_groups(cfg.config)
        series = group_series(cfg.results_dir, group, members)
        isempty(series) && continue
        fig = weak_scaling_figure(
            series; plot_title=group_title(group) * " — weak scaling"
        )
        out = joinpath(cfg.out_dir, "$(group)_weak_scaling$(cfg.output_suffix).png")
        savefig(fig, out)
        println("wrote $out")
    end
    return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
