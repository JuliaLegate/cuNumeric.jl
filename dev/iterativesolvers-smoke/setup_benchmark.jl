# Use an environment outside the repository's Julia workspace.
using Pkg
length(ARGS) == 1 || error("Usage: julia setup_benchmark.jl /path/to/benchmark-env")
Pkg.activate(abspath(ARGS[1]))
Pkg.develop(path=normpath(joinpath(@__DIR__, "../..")))
Pkg.add([
    PackageSpec(name="Krylov", version="0.10.10"),
    PackageSpec(name="CUDA", version="6.4.0"),
    PackageSpec(url="https://github.com/JuliaParallel/Dagger.jl.git",
                rev="ee7fcb13a21252878d738290c85af57de2824b7b"),
])
Pkg.precompile()
Pkg.status()
