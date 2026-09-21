using Pkg

# Run this script with --project=/path/to/a/separate/environment.
# No solver dependency is added to cuNumeric's own Project.toml.
source = abspath(get(ENV, "CUNUMERIC_DEV_PATH", joinpath(@__DIR__, "../..")))
normpath(dirname(Base.active_project())) == source &&
    error("Select a separate environment with --project=/path/to/solver-environment")
Pkg.develop(path=source)
Pkg.add(PackageSpec(name="IterativeSolvers", version="0.9.4"))
Pkg.status()
