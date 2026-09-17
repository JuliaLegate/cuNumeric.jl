# Standalone build regression tests; no CUDA, Legate, or network required.
# Run: julia --startup-file=no test/build_cxxwrap.jl
using Test
include(joinpath(@__DIR__, "..", "deps", "cxxwrap.jl"))
is_supported_version(v::VersionNumber) = v"26.6.0" <= v <= v"26.11.999"

@testset "libcxxwrap build cache recovery" begin
    mktempdir() do tmp
        root = replace(tmp, '\\' => '/')
        override = joinpath(root, "dev", "libcxxwrap_julia_jll", "override")
        mkpath(override)
        headers = "$root/headers"
        mkpath(headers)
        library = "$root/libcxxwrap.so"
        write(library, "fixture")
        config = """
        foreach(name cxxwrap_julia cxxwrap_julia_stl)
            add_library(JlCxx::\${name} SHARED IMPORTED)
            set_target_properties(JlCxx::\${name} PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "$headers;"
                IMPORTED_CONFIGURATIONS RELEASE
                IMPORTED_LOCATION_RELEASE "$library")
        endforeach()
        """
        config_path = joinpath(override, "JlCxxConfig.cmake")
        write(config_path, config)
        usable = cxxwrap_usable(override; log_dir=root)
        usable || print(read(joinpath(root, "libcxxwrap_check.log"), String))
        @test usable
        rm(headers; recursive=true)
        @test !cxxwrap_usable(override; log_dir=root)
        mkpath(headers)
        rm(library)
        @test !cxxwrap_usable(override; log_dir=root)
        write(library, "fixture")

        marker = joinpath(override, "LEGATE_INSTALL.txt")
        julia_marker = joinpath(override, "JULIA_INSTALL.txt")
        write(marker, "26.6.0")
        write(julia_marker, cxxwrap_julia_identity())
        original_depots = copy(DEPOT_PATH)
        try
            empty!(DEPOT_PATH)
            push!(DEPOT_PATH, root)
            # A healthy cache must work even with no installer present.
            @test isnothing(ensure_cxxwrap(root, v"26.6.0"; log_dir=root))
            scripts = joinpath(root, "scripts")
            mkpath(scripts)
            installer = joinpath(scripts, "install_cxxwrap.sh")
            write(installer, "#!/bin/bash\nexit 7\n")
            write(config_path, replace(config, headers => "$root/deleted-headers"))
            @test_throws ErrorException ensure_cxxwrap(root, v"26.6.0"; log_dir=root)
            @test !isfile(marker)
            @test !isfile(julia_marker)

            # Simulate successful rebuilding of the stale CMake export.
            write(joinpath(scripts, "JlCxxConfig.cmake"), config)
            write(installer, "#!/bin/bash\ncp \"\$1/scripts/JlCxxConfig.cmake\" \"\$1/dev/libcxxwrap_julia_jll/override/JlCxxConfig.cmake\"\n")
            @test isnothing(ensure_cxxwrap(root, v"26.6.0"; log_dir=root))
            @test read(marker, String) == "26.6.0"
            @test read(julia_marker, String) == cxxwrap_julia_identity()

            # Existing paths and a matching provider version are insufficient
            # when Julia changes, or when a legacy build has no Julia marker.
            write(installer, "#!/bin/bash\nexit 7\n")
            old_version = VERSION.major == 1 && VERSION.minor == 11 ? "1.12.0" : "1.11.0"
            write(julia_marker, old_version * "\n" * realpath(joinpath(Sys.BINDIR, Base.julia_exename())))
            @test_throws ErrorException ensure_cxxwrap(root, v"26.6.0"; log_dir=root)
            @test !isfile(marker)
            @test !isfile(julia_marker)
            write(marker, "26.6.0")
            @test_throws ErrorException ensure_cxxwrap(root, v"26.6.0"; log_dir=root)
            @test !isfile(marker)

            # Moving to another Julia installation also forces a rebuild.
            write(marker, "26.6.0")
            write(julia_marker, string(VERSION, "\n/old/julia/bin/julia"))
            @test_throws ErrorException ensure_cxxwrap(root, v"26.6.0"; log_dir=root)
            @test !isfile(julia_marker)

            # A zero exit status without usable outputs is not success.
            write(config_path, replace(config, headers => "$root/deleted-headers"))
            write(installer, "#!/bin/bash\nexit 0\n")
            @test_throws ErrorException ensure_cxxwrap(root, v"26.6.0"; log_dir=root)
            @test !isfile(marker)
            @test !isfile(julia_marker)
        finally
            empty!(DEPOT_PATH)
            append!(DEPOT_PATH, original_depots)
        end
    end
end
