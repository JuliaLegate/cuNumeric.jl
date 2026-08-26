using Documenter, DocumenterVitepress
using cuNumeric
using CNPreferences
using Random
using TOML

ci = get(ENV, "CI", "") == "true"

# Read cupynumeric_jll compat from the package Project.toml so conda docs stay in sync.
const CUPYNUMERIC_JLL_COMPAT, CUPYNUMERIC_MAJOR_MINOR = let
    project_toml = joinpath(@__DIR__, "..", "Project.toml")
    compat = String(TOML.parsefile(project_toml)["compat"]["cupynumeric_jll"])
    m = match(r"(\d+)\.(\d+)", compat)
    m === nothing && error("Could not parse major.minor from cupynumeric_jll compat '$compat'")
    compat, "$(m.captures[1]).$(m.captures[2])"
end
@info "Docs using cupynumeric_jll compat=$CUPYNUMERIC_JLL_COMPAT (conda line $CUPYNUMERIC_MAJOR_MINOR)"

# DocumenterVitepress hardcodes nested sidebar groups as collapsed: false.
# Prefer closed-by-default dropdowns in the left nav.
@eval DocumenterVitepress function pagelist2str(doc, name_contents::Pair{String,<:AbstractVector})
    name, contents = name_contents
    rendered_contents = pagelist2str.((doc,), contents)
    return "{ text: '$(replace(name, "'" => "\\'"))', collapsed: true, items: [\n$(join(rendered_contents, ",\n"))]\n }"
end

makedocs(;
    sitename="cuNumeric.jl",
    authors="Ethan Meitz and David Krasowska",
    format=MarkdownVitepress(;
        repo="github.com/JuliaLegate/cuNumeric.jl",
        devbranch="main",
        devurl="dev",
    ),
    pages=[
        "Home" => "index.md",
        "Examples" => [
            "Initialization" => "examples/initialization.md",
            "Monte-Carlo" => "examples/montecarlo.md",
            "Gray-Scott" => "examples/grayscott.md",
            "Dynamic Mode Decomposition" => "examples/dmd.md",
            "Periodic Poisson (FFT)" => "examples/poisson_fft.md",
            "Tensor Network Contraction" => "examples/tensor_network.md",
        ],
        "Performance Tips" => [
            "Kernel Fusion" => "perf/kernel_fusion.md",
            "The @accelerate Macro" => "perf/reduce_allocations.md",
            "Patterns to Avoid" => "perf/patterns_to_avoid.md",
        ],
        "Configuration" => [
            "Hardware" => "configuration/hardware.md",
            "Build Modes" => "install.md",
            "CNPreferences" => "api_preferences.md",
        ],
        "Benchmarks" => [
            "Results" => "benchmarks/results.md",
            "How to Benchmark" => "benchmarks/howto.md",
        ],
        "Developer" => [
            "Developer Mode" => "developer_mode.md",
            "Debugging" => "debugging.md",
            "Internals" => "internals.md",
        ],
        "Public API" => [
            "Initialization" => "api_initialization.md",
            "Random" => "api_random.md",
            "Unary Operations" => "api_unary.md",
            "Binary Operations" => "api_binary.md",
            "Linear Algebra" => "linalg.md",
            "FFT" => "fft.md",
            "HDF5" => "api_hdf5.md",
            "NDArray Reference" => "api.md",
            "CUDA.jl Tasking" => "api_cuda.md",
            "Internal API" => "api_internal.md",
        ],
    ],
)

if ci
    @info "Deploying Docs to GitHub Pages"
    DocumenterVitepress.deploydocs(;
        repo="github.com/JuliaLegate/cuNumeric.jl",
        target=joinpath(@__DIR__, "build"),
        branch="gh-pages",
        devbranch="main",
        push_preview=true,
    )
end
