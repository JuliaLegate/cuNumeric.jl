using Documenter, DocumenterVitepress

makedocs(;
    sitename="cuNumeric.jl",
    authors="Ethan Meitz and David Krasowska",
    format=MarkdownVitepress(;
        repo="github.com/JuliaLegate/cuNumeric.jl.git",
        devbranch="main",
        devurl="dev",
        deploy_url="Julialegate.github.io/cuNumeric.jl/",
    ),
    pages=[
        "Home" => "index.md",
        "Build Options" => "install.md",
        "Examples" => "examples.md",
        "Performance Tips" => "perf.md",
        "Back End Details" => "usage.md",
        "Benchmarking" => "benchmark.md",
        "Public API" => "api.md",
    ],
)

DocumenterVitepress.deploydocs(;
    repo="github.com/JuliaLegate/cuNumeric.jl.git",
    target=joinpath(@__DIR__, "build"),
    branch="gh-pages",
    devbranch="main",
    push_preview=true,
)
