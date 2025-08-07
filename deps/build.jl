#= Copyright 2025 Northwestern University, 
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
using CNPreferences: CNPreferences

const SUPPORTED_CUPYNUMERIC_VERSIONS = ["25.05.00"]
const LATEST_CUPYNUMERIC_VERSION = SUPPORTED_CUPYNUMERIC_VERSIONS[end]

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
        println(contents)
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

function build_jlcxxwrap(repo_root)
    @info "libcxxwrap: Downloading"
    build_libcxxwrap = joinpath(repo_root, "scripts/install_cxxwrap.sh")

    # this is actually correct even for cunumeric.
    version_path = joinpath(DEPOT_PATH[1], "dev/libcxxwrap_julia_jll/override/LEGATE_INSTALL.txt")

    if isfile(version_path)
        version = strip(read(version_path, String))
        if version ∈ SUPPORTED_CUPYNUMERIC_VERSIONS
            @info "libcxxwrap: Found supported version built with Legate.jl: $version"
            return nothing
        else
            @info "libcxxwrap: Unsupported version found: $version. Rebuilding..."
        end
    else
        @info "libcxxwrap: No version file found. Starting build..."
    end

    @info "libcxxwrap: Running build script: $build_libcxxwrap"
    run_sh(`bash $build_libcxxwrap $repo_root`, "libcxxwrap")
    open(version_path, "w") do io
        write(io, LATEST_CUPYNUMERIC_VERSION)
    end
end

function build_cpp_wrapper(
    repo_root, cupynumeric_loc, legate_loc, hdf5_root, blas_lib, install_root
)
    @info "libcunumeric_jl_wrapper: Building C++ Wrapper Library"
    if isdir(install_root)
        @warn "libcunumeric_jl_wrapper: Build dir exists. Deleting prior build."
        rm(install_root; recursive=true)
        mkdir(install_root)
    end

    branch = load_preference(
        CNPreferences, "wrapper_branch", CNPreferences.DEVEL_DEFAULT_WRAPPER_BRANCH
    )

    build_cpp_wrapper = joinpath(repo_root, "scripts/build_cpp_wrapper.sh")
    nthreads = Threads.nthreads()
    run_sh(
        `bash $build_cpp_wrapper $repo_root $cupynumeric_loc $legate_loc $hdf5_root $blas_lib $install_root $branch $nthreads`,
        "cpp_wrapper",
    )
end

function build(mode)
    if mode == CNPreferences.MODE_JLL
        @warn "No reason to Build on JLL mode."
    end
    pkg_root = abspath(joinpath(@__DIR__, "../"))
    deps_dir = joinpath(@__DIR__)

    build_log = joinpath(deps_dir, "build.log")
    open(build_log, "w") do io
        println(io, "=== Build started ===")
    end

    @info "cuNumeric.jl: Parsed Package Dir as: $(pkg_root)"

    hdf5_lib = Legate.get_install_libhdf5()
    legate_lib = Legate.get_install_liblegate()
    cupynumeric_lib = load_preference(CNPreferences, "CUPYNUMERIC_LIB", nothing)
    blas_lib = load_preference(CNPreferences, "BLAS_LIB", nothing)

    if mode == CNPreferences.MODE_DEVELOPER
        install_lib = joinpath(pkg_root, "deps", "cunumeric_jl_wrapper")
        build_jlcxxwrap(pkg_root)
        build_cpp_wrapper(
            pkg_root, up_dir(cupynumeric_lib), up_dir(legate_lib), up_dir(hdf5_lib), blas_lib,
            install_lib,
        )
    end
end

const JULIA_LEGATE_BUILDING_DOCS = get(ENV, "JULIA_LEGATE_BUILDING_DOCS", "false") == "true"
const mode = load_preference(CNPreferences, "mode", CNPreferences.MODE_JLL)

if !JULIA_LEGATE_BUILDING_DOCS
    build(mode)
end
