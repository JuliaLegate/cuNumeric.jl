# Standalone build regression test; no CUDA, Legate, or network required.
# Run: julia --startup-file=no test/build_cxxwrap.jl
using Test
include(joinpath(@__DIR__, "..", "deps", "cxxwrap.jl"))

@testset "libcxxwrap cache validation" begin
    mktempdir() do tmp
        root = replace(tmp, '\\' => '/')
        headers = "$root/headers"
        library = "$root/libcxxwrap.so"
        override = "$root/override"
        mkpath(override)
        mkpath(headers)
        write(library, "fixture")
        write(
            joinpath(override, "JlCxxConfig.cmake"),
            """
foreach(name cxxwrap_julia cxxwrap_julia_stl)
    add_library(JlCxx::\${name} SHARED IMPORTED)
    set_target_properties(JlCxx::\${name} PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "$headers"
        IMPORTED_LOCATION "$library")
endforeach()
""",
        )

        @test cxxwrap_usable(override; log_dir=root)
        rm(headers; recursive=true)
        @test !cxxwrap_usable(override; log_dir=root)
    end
end
