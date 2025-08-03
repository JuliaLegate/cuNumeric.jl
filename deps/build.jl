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
using Pkg
using OpenSSL_jll
using HDF5_jll
using NCCL_jll
using CUTENSOR_jll
using cupynumeric_jll
using Preferences
using Legate

include(joinpath(@__DIR__, "..", "src", "develop", "cunumeric_wrapper.jl"))

const SUPPORTED_CUPYNUMERIC_VERSIONS = ["25.05.00"]
const LATEST_CUPYNUMERIC_VERSION = SUPPORTED_CUPYNUMERIC_VERSIONS[end]

# Automatically pipes errors to new file
# and appends stdout to build.log
function run_sh(cmd::Cmd, filename::String)
    println(cmd)

    build_log = joinpath(@__DIR__, "build.log")
    err_log = joinpath(@__DIR__, "$(filename).err")

    if isfile(err_log)
        rm(err_log)
    end

    try
        run(pipeline(cmd; stdout=build_log, stderr=err_log, append=false))
    catch e
        println("stderr log generated: ", err_log, '\n')
        exit(-1)
    end
end

function get_library_root(jll_module, env_var::String)
    if haskey(ENV, env_var)
        return get(ENV, env_var, "0")
    elseif jll_module.is_available()
        return joinpath(jll_module.artifact_dir, "lib")
    else
        error("$env_var not found via environment or JLL.")
    end
end

function build_cpp_wrapper(repo_root, cupynumeric_loc, legate_loc, hdf5_lib)
    @info "libcupynumericwrapper: Building C++ Wrapper Library"
    install_dir = joinpath(repo_root, "deps", "cunumeric_wrapper_install")
    if isdir(install_dir)
        @warn "libcunumericwrapper: Build dir exists. Deleting prior build."
        rm(install_dir; recursive=true)
        mkdir(install_dir)
    end

    build_cpp_wrapper = joinpath(repo_root, "scripts/build_cpp_wrapper.sh")
    nthreads = Threads.nthreads()
    run_sh(
        `bash $build_cpp_wrapper $repo_root $cupynumeric_loc $legate_loc $hdf5_lib $install_dir $nthreads`,
        "cpp_wrapper",
    )
end

function is_cupynumeric_installed(cupynumeric_root::String; throw_errors::Bool=false)
    include_dir = joinpath(cupynumeric_root, "include")
    if !isdir(joinpath(include_dir, "cupynumeric"))
        throw_errors &&
            @error "cuNumeric.jl: Cannot find include/cupynumeric in $(cupynumeric_root)"
        return false
    end
    return true
end

function parse_cupynumeric_version(cupynumeric_root)
    version_file = joinpath(cupynumeric_root, "include", "cupynumeric", "version_config.hpp")

    version = nothing
    open(version_file, "r") do f
        data = readlines(f)
        major = parse(Int, split(data[end - 2])[end])
        minor = lpad(split(data[end - 1])[end], 2, '0')
        patch = lpad(split(data[end])[end], 2, '0')
        version = "$(major).$(minor).$(patch)"
    end

    if isnothing(version)
        error("cuNumeric.jl: Failed to parse version from conda environment")
    end

    return version
end

function install_cupynumeric(repo_root, version_to_install)
    @info "libcupynumeric: Building cupynumeric"

    build_dir = joinpath(repo_root, "libcupynumeric")
    if !isdir(build_dir)
        mkdir(build_dir)
    else
        @warn "libcupynumeric: Build dir exists. Deleting prior build."
        rm(build_dir; recursive=true)
        mkdir(build_dir)
    end

    legate_root = joinpath(Legate.get_install_liblegate(), "..") # new gives /lib

    nccl_root = joinpath(Legate.get_install_libnccl(), "..")
    cutensor_root = joinpath(get_library_root(CUTENSOR_jll, "JULIA_CUTENSOR_PATH"), "..")

    build_cupynumeric = joinpath(repo_root, "scripts/build_cupynumeric.sh")
    nthreads = Threads.nthreads()
    run_sh(
        `bash $build_cupynumeric $repo_root $legate_root $nccl_root $cutensor_root $build_dir $version_to_install $nthreads`,
        "cupynumeric",
    )
end

function check_prefix_install(env_var, env_loc)
    if get(ENV, env_var, "0") == "1"
        @info "cuNumeric.jl: Using $(env_var) mode"
        cupynumeric_root = get(ENV, env_loc, nothing)
        cupynumeric_installed = is_cupynumeric_installed(cupynumeric_root)
        if !cupynumeric_installed
            error("cuNumeric.jl: Build halted: cupynumeric not found in $cupynumeric_root")
        end
        installed_version = parse_cupynumeric_version(cupynumeric_root)
        if installed_version ∉ SUPPORTED_CUPYNUMERIC_VERSIONS
            error(
                "cuNumeric.jl: Build halted: $(cupynumeric_root) detected unsupported version $(installed_version)"
            )
        end
        @info "cuNumeric.jl: Found a valid install in: $(cupynumeric_root)"
        return true
    end
    return false
end

function build()
    pkg_root = abspath(joinpath(@__DIR__, "../"))
    deps_dir = joinpath(@__DIR__)

    @info "cuNumeric.jl: Parsed Package Dir as: $(pkg_root)"
    # custom install 
    if check_prefix_install("CUNUMERIC_CUSTOM_INSTALL", "CUNUMERIC_CUSTOM_INSTALL_LOCATION")
        cupynumeric_root = get(ENV, "CUNUMERIC_CUSTOM_INSTALL_LOCATION", nothing)
        # conda install 
    elseif check_prefix_install("CUNUMERIC_LEGATE_CONDA_INSTALL", "CONDA_PREFIX")
        cupynumeric_root = get(ENV, "CONDA_PREFIX", nothing)
    elseif get(ENV, "CUNUMERIC_INSTALL_CUPYNUMERIC", "0") == "1"
        cupynumeric_root = abspath(joinpath(@__DIR__, "../libcupynumeric"))
        cupynumeric_installed = is_cupynumeric_installed(cupynumeric_root)
        if cupynumeric_installed
            installed_version = parse_cupynumeric_version(cupynumeric_root)
            if installed_version ∉ SUPPORTED_CUPYNUMERIC_VERSIONS
                @warn "cuNumeric.jl: Detected unsupported version of cupynumeric installed: $(installed_version). Installing newest version."
                install_cupynumeric(pkg_root, LATEST_CUPYNUMERIC_VERSION)
            else
                @info "cuNumeric.jl: Found cupynumeric already installed."
            end
        else
            install_cupynumeric(pkg_root, LATEST_CUPYNUMERIC_VERSION)
        end
    else # default
        cupynumeric_root = cupynumeric_jll.artifact_dir
    end

    legate_lib = Legate.get_install_liblegate()
    legate_root = joinpath(legate_lib, "..")

    cupynumeric_lib = joinpath(cupynumeric_root, "lib")
    if haskey(ENV, "JULIA_TBLIS_PATH")
        tblis_lib = get(ENV, "JULIA_TBLIS_PATH", "0")
    else
        tblis_lib = joinpath(cupynumeric_root, "lib")
    end

    hdf5_lib = Legate.get_install_libhdf5()
    cutensor_lib = get_library_root(CUTENSOR_jll, "JULIA_CUTENSOR_PATH")

    if get(ENV, "CUNUMERIC_DEVELOP_MODE", "0") == "1"
        # create libcupynumericwrapper.so
        build_cpp_wrapper(pkg_root, cupynumeric_root, legate_root, hdf5_lib)
        cunumeric_wrapper_lib = joinpath(pkg_root, "deps", "cunumeric_wrapper_install")
    elseif true == true # temporary until cunumeric_jl_wrapper_jll
        cunumeric_wrapper_lib = cunumeric_wrapper_jll_local_branch_install()
    else
        cunumeric_wrapper_lib = joinpath(cunumeric_jl_wrapper_jll.artifact_dir, "lib")
    end

    open(joinpath(deps_dir, "deps.jl"), "w") do io
        println(io, "const CUTENSOR_LIB = \"$(cutensor_lib)\"")
        println(io, "const TBLIS_LIB = \"$(tblis_lib)\"")
        println(io, "const CUPYNUMERIC_LIB = \"$(cupynumeric_lib)\"")
        println(io, "const CUNUMERIC_WRAPPER_LIB = \"$(cunumeric_wrapper_lib)\"")
    end
end

const JULIA_LEGATE_BUILDING_DOCS = get(ENV, "JULIA_LEGATE_BUILDING_DOCS", "false") == "true"
if !JULIA_LEGATE_BUILDING_DOCS
    build()
end
