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

using Pkg
using Preferences
using Legate
using CNPreferences
using CUDACore: CUDACore

# Maybe needed as build deps
using cupynumeric_jll: cupynumeric_jll
using OpenBLAS32_jll: OpenBLAS32_jll

const BuildTools = Legate.BuildTools

include("version.jl")

function build_cpp_wrapper(
    repo_root, cupynumeric_loc, legate_loc, blas_loc, install_root;
    cuda_root=nothing, cuda_enabled=true,
)
    @info "libcunumeric_jl_wrapper: Building C++ Wrapper Library"
    isdir(install_root) && (rm(install_root; recursive=true); mkdir(install_root))
    bld_command = `$(joinpath(repo_root, "scripts/build_cpp_wrapper.sh")) $repo_root $cupynumeric_loc $legate_loc $blas_loc $install_root $(Threads.nthreads())`
    BuildTools.run_build_wrapper_script(
        repo_root, bld_command; cuda_root, cuda_enabled, log_dir=@__DIR__
    )
end

function build_deps(pkg_root, cupynumeric_root, blas_root; cuda_root=nothing, cuda_enabled=true)
    BuildTools.check_cmake_version(Legate.MIN_CMAKE_VERSION)
    legate_lib = Legate.get_install_liblegate()
    install_lib = joinpath(pkg_root, "lib", "cunumeric_jl_wrapper", "build")
    if !cupynumeric_valid(cupynumeric_root)
        error(
            "cuNumeric.jl: Unsupported cuNumeric version at $(cupynumeric_root). " *
            "Installed version: $(get_cupynumeric_version(cupynumeric_root)) not in range supported: " *
            "$(MIN_CUNUMERIC_VERSION)-$(MAX_CUNUMERIC_VERSION).",
        )
    end

    BuildTools.build_jlcxxwrap(
        pkg_root, get_cupynumeric_version(cupynumeric_root);
        log_dir=@__DIR__, is_compatible=is_supported_version,
    )
    build_cpp_wrapper(
        pkg_root, cupynumeric_root, up_dir(legate_lib), blas_root,
        install_lib;
        cuda_root, cuda_enabled,
    )
    BuildTools.set_jll_artifact_override(:cunumeric_jl_wrapper_jll, install_lib)
end

function build(::CNPreferences.JLL)
    @warn "No reason to Build on JLL mode. Exiting Build"
    return nothing
end

function build(::CNPreferences.Conda)
    @warn "Conda Build does not currently pass our CI. Proceed with caution."
    pkg_root = BuildTools.start_build("cuNumeric.jl", @__DIR__)

    cupynumeric_root = load_preference(CNPreferences, "cunumeric_conda_env", nothing)
    cuda_toolkit_root = load_preference(CNPreferences, "CUDA_TOOLKIT_ROOT", nothing)
    if isnothing(cupynumeric_root)
        error("This shouldn't happen. cunumeric_conda_env = nothing?")
    end
    if isnothing(cuda_toolkit_root)
        error(
            "CUDA_TOOLKIT_ROOT must be set by CNPreferences to point to the CUDA linked in your cupynumeric build."
        )
    end

    #!TODO SET LocalPreferences.toml to use local CUDA libraries

    is_cupynumeric_installed(cupynumeric_root; throw_errors=true)
    build_deps(pkg_root, cupynumeric_root, cupynumeric_root)
end

function build(::CNPreferences.Developer)
    pkg_root = BuildTools.start_build("cuNumeric.jl", @__DIR__)

    cupynumeric_root = load_preference(CNPreferences, "cunumeric_path", nothing)
    blas_lib = load_preference(CNPreferences, "BLAS_LIB", nothing)

    if isnothing(cupynumeric_root)
        cupynumeric_root, cuda_root = BuildTools.setup_jll_build_env(
            pkg_root, BuildTools.CUNUMERIC_JLL_DEP
        )
        cuda_enabled = !isnothing(cuda_root) # cuda_root resolving to nothing means there is no cuda
    else
        is_cupynumeric_installed(cupynumeric_root; throw_errors=true)
        cuda_enabled, cuda_root = BuildTools.resolve_custom_cuda("cupynumeric") # cuda_root is nothing.
    end

    blas_lib = something(blas_lib, BuildTools.find_jll_artifact_dir(:OpenBLAS32_jll))
    build_deps(pkg_root, cupynumeric_root, up_dir(blas_lib); cuda_root, cuda_enabled)
end

const mode_str = load_preference(CNPreferences, "cunumeric_mode", CNPreferences.MODE_JLL)
build(CNPreferences.to_mode(mode_str))
