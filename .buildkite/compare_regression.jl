# Compare benchmark harness results from the base branch and this commit.
#
#     julia compare_regression.jl <base_root> <candidate_root> [--threshold=10] [--base=main] [--out=report.md]
#
# Each root holds the harness run directory for that side. Exits 1 on a
# slowdown above the threshold or a failed candidate run; base failures are
# only reported so a PR can fix them.

using Printf
using Statistics
using TOML

function load_runs(root)
    runs = Dict{Tuple,Union{Float64,Nothing}}()
    isdir(root) || return runs
    for dir in readdir(root; join=true)
        manifest_path = joinpath(dir, "manifest.toml")
        isfile(manifest_path) || continue
        for r in TOML.parsefile(manifest_path)["runs"]
            key = (r["name"], r["T"], r["N"], r["M"])
            runs[key] = r["status"] == "complete" ? median_time(dir, r) : nothing
        end
    end
    return runs
end

# Median trial time in ms, or `nothing` when rows are missing or incorrect.
function median_time(dir, r)
    # The harness writes `<name>_<save_as>.csv`, where save_as extends the model
    # (e.g. `cunumeric_nofusion`, `cunumeric_struct`).
    results = joinpath(dir, r["results_subdir"])
    isdir(results) || return nothing
    prefix = "$(r["name"])_$(r["model"])"
    files = filter(readdir(results)) do f
        return f == "$prefix.csv" || (startswith(f, "$(prefix)_") && endswith(f, ".csv"))
    end
    times = Float64[]
    for path in joinpath.(results, files), line in eachline(path)
        f = split(strip(line), ',')
        length(f) == 8 || continue
        (parse(Int, f[3]), parse(Int, f[4])) == (r["N"], r["M"]) || continue
        f[8] == "fail" && return nothing
        push!(times, parse(Float64, f[6]))
    end
    return isempty(times) ? nothing : median(times)
end

label(key) = "$(key[1]) ($(key[2]), $(key[3])×$(key[4]))"

function report(base_root, candidate_root, threshold, base_name)
    base = load_runs(base_root)
    candidate = load_runs(candidate_root)
    isempty(candidate) && error("no candidate results in $candidate_root")

    lines = [
        "## Benchmark regression vs. `$base_name`", "",
        "| Benchmark | `$base_name` (ms) | PR (ms) | Change |",
        "| --- | ---: | ---: | ---: |",
    ]
    regressions, failed, uncompared = String[], String[], String[]
    for key in sort!(collect(keys(candidate)))
        after = candidate[key]
        before = get(base, key, nothing)
        if after === nothing
            push!(failed, label(key))
        elseif before === nothing
            push!(uncompared, label(key))
        else
            change = 100 * (after / before - 1)
            flag = change > threshold ? " ⚠️" : ""
            push!(
                lines,
                @sprintf("| %s | %.3f | %.3f | %+.1f%%%s |",
                    label(key), before, after, change, flag)
            )
            change > threshold &&
                push!(regressions, @sprintf("%s: %.1f%% slower", label(key), change))
        end
    end
    for (title, items) in (("Slower than the threshold", regressions),
        ("Failed on this PR", failed),
        ("Not compared (no base result)", uncompared))
        isempty(items) && continue
        append!(lines, ["", "**$title:**"], ["- $item" for item in items])
    end
    push!(lines, "", @sprintf("Threshold: more than %.0f%% slower (median of trials).", threshold))
    return join(lines, '\n') * '\n', isempty(regressions) && isempty(failed)
end

function main(args)
    threshold, out, base_name = 10.0, nothing, "base"
    positional = String[]
    for arg in args
        if startswith(arg, "--threshold=")
            threshold = parse(Float64, split(arg, '='; limit=2)[2])
        elseif startswith(arg, "--base=")
            base_name = split(arg, '='; limit=2)[2]
        elseif startswith(arg, "--out=")
            out = split(arg, '='; limit=2)[2]
        else
            push!(positional, arg)
        end
    end
    length(positional) == 2 || error("usage: compare_regression.jl <base_root> <candidate_root>")
    text, ok = report(positional..., threshold, base_name)
    print(text)
    out === nothing || write(out, text)
    return ok ? 0 : 1
end

try
    exit(main(ARGS))
catch e
    println(stderr, "comparison failed: ", sprint(showerror, e))
    exit(2)
end
