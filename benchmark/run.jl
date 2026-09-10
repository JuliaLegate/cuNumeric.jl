# Orchestrator remains off the GPU; each timed backend uses its own process.
using Pkg

function ensure_project_ready()
    Pkg.develop([
        Pkg.PackageSpec(; path=joinpath(@__DIR__, "..", "lib", "CNPreferences")),
        Pkg.PackageSpec(; path=joinpath(@__DIR__, "..")),
    ])
    Pkg.instantiate()
end

function cupynumeric_env_name()
    haskey(ENV, "CUPYNUMERIC_ENV") && return ENV["CUPYNUMERIC_ENV"]
    for (_, info) in Pkg.dependencies()
        info.name == "cupynumeric_jll" || continue
        info.version === nothing && continue
        return "cupynumeric-bench-$(info.version.major).$(info.version.minor)"
    end
    return error("could not resolve cupynumeric_jll version; set CUPYNUMERIC_ENV explicitly")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    ensure_project_ready()
    include("src/core.jl")
    include_benchmarks()
    include("src/parse_benchmarks.jl")
    include("src/memory.jl")
    include("src/planning.jl")
    include("src/runner.jl")
    using CNPreferences: CNPreferences
    exit(main())
end
