#= Copyright 2026 Northwestern University, 
 *                   Carnegie Mellon University University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author(s): David Krasowska <krasow@u.northwestern.edu>
 *            Ethan Meitz <emeitz@andrew.cmu.edu>
=#

using Preferences
using Legate
using CNPreferences

include("version.jl")

up_dir(dir::String) = abspath(joinpath(dir, ".."))

# Automatically pipes errors to new file
# and appends stdout to build.log
function run_sh(cmd::Cmd, filename::String)
    println(cmd)

    build_log = joinpath(@__DIR__, "build.log")
    tmp_build_log = joinpath(@__DIR__, "$(filename).log")
    err_log = joinpath(@__DIR__, "$(filename).err")

    if isfile(err_log)
        rm(err_log)
    end

    if isfile(tmp_build_log)
        rm(tmp_build_log)
    end

    try
        run(pipeline(cmd; stdout=tmp_build_log, stderr=err_log, append=false))
        contents = read(tmp_build_log, String)
        open(build_log, "a") do io
            println(contents)
        end
    catch e
        println("stderr log generated: ", err_log, '\n')
        contents = read(err_log, String)
        if !isempty(strip(contents))
            println("---- Begin stderr log ----")
            println(contents)
            println("---- End stderr log ----")
        end
    end
end

function build_jlcxxwrap(repo_root, cupynumeric_root)
    build_libcxxwrap = joinpath(repo_root, "scripts/install_cxxwrap.sh")
    version_path = joinpath(DEPOT_PATH[1], "dev/libcxxwrap_julia_jll/override/LEGATE_INSTALL.txt")
    if isfile(version_path)
        version = VersionNumber(strip(read(version_path, String)))
        @info "libcxxwrap: Found cuNumeric $version"
        if is_supported_version(version)
            @info "libcxxwrap: Found supported version built with cuNumeric.jl: $version"
            return nothing
        else
            @info "libcxxwrap: Unsupported version found: $version. Rebuilding..."
        end
    else
        @info "libcxxwrap: No version file found. Starting build..."
    end

    @info "libcxxwrap: Running build script: $build_libcxxwrap"
    run_sh(`bash $build_libcxxwrap $repo_root`, "libcxxwrap")
    mkpath(dirname(version_path))
    open(version_path, "w") do io
        write(io, string(get_cupynumeric_version(cupynumeric_root)))
    end
end

function build_cpp_wrapper(
    repo_root, cupynumeric_loc, legate_loc, blas_loc, install_root
)
    @info "libcunumeric_jl_wrapper: Building C++ Wrapper Library"
    if isdir(install_root)
        rm(install_root; recursive=true)
        mkdir(install_root)
    end

    build_cpp_wrapper = joinpath(repo_root, "scripts/build_cpp_wrapper.sh")
    nthreads = Threads.nthreads()

    bld_command = `$build_cpp_wrapper $repo_root $cupynumeric_loc $legate_loc $blas_loc $install_root $nthreads`

    # write out a bash script for debugging
    cmd_str = join(bld_command.exec, " ")
    wrapper_path = joinpath(repo_root, "build_wrapper.sh")
    open(wrapper_path, "w") do io
        println(io, "#!/bin/bash")
        println(io, "set -xe")
        println(io, cmd_str)
    end
    chmod(wrapper_path, 0o755)

    @info "Running build command: $bld_command"
    run_sh(`bash $bld_command`, "cpp_wrapper")
end

function _find_jll_artifact_dir(jll)
    eval(:(using $(jll)))
    jll_mod = getfield(Main, jll)
    root = jll_mod.artifact_dir
    return root
end

function _start_build()
    pkg_root = up_dir(@__DIR__)
    deps_dir = joinpath(@__DIR__)

    build_log = joinpath(deps_dir, "build.log")
    open(build_log, "w") do io
        println(io, "=== Build started ===")
    end

    @info "cuNumeric.jl: Parsed Package Dir as: $(pkg_root)"
    return pkg_root
end

"""
    build CxxWrap and cunumeric_jl_wrapper
"""
function build_deps(pkg_root, cupynumeric_root)
    legate_lib = Legate.get_install_liblegate()
    install_lib = joinpath(pkg_root, "lib", "cunumeric_jl_wrapper", "build")
    if !cupynumeric_valid(cupynumeric_root)
        error(
            "cuNumeric.jl: Unsupported cuNumeric version at $(cupynumeric_root). " *
            "Installed version: $(get_cupynumeric_version(cupynumeric_root)) not in range supported: " *
            "$(MIN_CUNUMERIC_VERSION)-$(MAX_CUNUMERIC_VERSION).",
        )
    end
    build_jlcxxwrap(pkg_root, cupynumeric_root)
    build_cpp_wrapper(
        pkg_root, cupynumeric_root, up_dir(legate_lib),
        install_lib,
    ) # $pkg_root/lib/cunumeric_jl_wrapper
end

function build(::CNPreferences.JLL)
    @warn "No reason to Build on JLL mode. Exiting Build"
    return nothing
end

function build(::CNPreferences.Conda)
    @warn "Conda Build does not currently pass our CI. Proceed with caution."
    pkg_root = _start_build()

    cupynumeric_root = load_preference(CNPreferences, "cunumeric_conda_env", nothing)
    if isnothing(cupynumeric_root)
        error("This shouldn't happen. cunumeric_conda_env = nothing?")
    end

    is_cupynumeric_installed(cupynumeric_root; throw_errors=true)
    build_deps(pkg_root, cupynumeric_root)
end

function build(::CNPreferences.Developer)
    pkg_root = _start_build()

    # can be nothing so this errors if not set
    cupynumeric_root = load_preference(CNPreferences, "cunumeric_path", nothing)
    if isnothing(cupynumeric_root)
        # we are using cupynumeric_jll
        cupynumeric_root = _find_jll_artifact_dir(:cupynumeric_jll)
    else
        # this means we have a custom path set
        is_cupynumeric_installed(cupynumeric_root; throw_errors=true)
    end

    build_deps(pkg_root, cupynumeric_root)
end

const mode_str = load_preference(CNPreferences, "cunumeric_mode", CNPreferences.MODE_JLL)
build(CNPreferences.to_mode(mode_str))
