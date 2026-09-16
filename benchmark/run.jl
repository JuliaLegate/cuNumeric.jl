# Orchestrator remains off the GPU; each timed backend uses its own process.
using Pkg, TOML

function ensure_project_ready()
    return Pkg.instantiate()
end

function cupynumeric_env_name()
    haskey(ENV, "CUPYNUMERIC_ENV") && return ENV["CUPYNUMERIC_ENV"]
    manifest = joinpath(@__DIR__, "environments", "cunumeric", "Manifest.toml")
    isfile(manifest) || error(
        "cuNumeric worker manifest is missing; instantiate environments/cunumeric " *
        "or set CUPYNUMERIC_ENV explicitly",
    )
    deps = get(TOML.parsefile(manifest), "deps", Dict{String,Any}())
    entries = get(deps, "cupynumeric_jll", Any[])
    isempty(entries) &&
        error("could not resolve cupynumeric_jll; set CUPYNUMERIC_ENV explicitly")
    entry = entries isa AbstractVector ? first(entries) : entries
    version = VersionNumber(entry["version"])
    return "cupynumeric-bench-$(version.major).$(version.minor)"
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    ensure_project_ready()
    include("src/core.jl")
    include_benchmarks()
    # cuNumeric-only accelerated variants; orchestrator needs the types for planning.
    include("src/cunumeric/benchmarks/grayscott_accelerate_forms.jl")
    include("src/models.jl")
    include("src/parse_benchmarks.jl")
    include("src/memory.jl")
    include("src/planning.jl")
    include("src/runner.jl")
    exit(main())
end
