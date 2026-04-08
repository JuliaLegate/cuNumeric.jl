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

function check_cupynumeric_install(cupynumeric_root)
    is_cupynumeric_installed(cupynumeric_root; throw_errors=true)
    if !cupynumeric_valid(cupynumeric_root)
        error(
            "cuNumeric.jl: Unsupported cuNumeric version at $(cupynumeric_root). " *
            "Installed version: $(get_cupynumeric_version(cupynumeric_root)) not in range supported: " *
            "$(MIN_CUNUMERIC_VERSION)-$(MAX_CUNUMERIC_VERSION).",
        )
    end

    @debug "cuNumeric.jl: Found a valid install in: $(cupynumeric_root)"
    return true
end

#############################################
# LOTS OF THIS LOGIC EXISTS IN LEGATE.JL TOO
# DEFINITELY DUPLICATED CODE
#############################################

function check_jll(m::Module)
    if !m.is_available()
        m_host_cuda = cupynumeric_jll.host_platform["cuda"]

        if (m_host_cuda == "none")
            error(
                "$(string(m)) installed but not available on this platform.\n $(string(cupynumeric_jll.host_platform))"
            )
        end

        v_host_cuda = VersionNumber(m_host_cuda)
        valid_cuda_version = Legate.MIN_CUDA_VERSION <= v_host_cuda <= Legate.MAX_CUDA_VERSION
        if !valid_cuda_version
            error(
                "$(string(m)) installed but not available on this platform. Host CUDA ver: $(v_host_cuda) not in range supported by $(string(m)): $(MIN_CUDA_VERSION)-$(MAX_CUDA_VERSION)."
            )
        else
            error("$(string(m)) installed but not available on this platform. Unknown reason.")
        end
    end
end

function find_paths(
    mode::String;
    cupynumeric_jll_module::Union{Module,Nothing}=nothing,
    cupynumeric_jll_wrapper_module::Union{Module,Nothing}=nothing,
)
    libcupynumeric_path, libcupynumeric_wrapper_path = cuNumeric._find_paths(
        CNPreferences.to_mode(mode), cupynumeric_jll_module, cupynumeric_jll_wrapper_module
    )
    set_preferences!(CNPreferences, "CUPYNUMERIC_LIBDIR" => libcupynumeric_path; force=true)
    set_preferences!(
        CNPreferences, "CUPYNUMERIC_WRAPPER_LIBDIR" => libcupynumeric_wrapper_path; force=true
    )
end

function _find_paths(
    mode::CNPreferences.JLL,
    cupynumeric_jll_module::Module,
    cupynumeric_jll_wrapper_module::Module,
)
    check_jll(cupynumeric_jll_module)
    check_jll(cupynumeric_jll_wrapper_module)
    cupynumeric_lib = joinpath(cupynumeric_jll_module.artifact_dir, "lib")
    wrapper_lib = joinpath(cupynumeric_jll_wrapper_module.artifact_dir, "lib")
    return cupynumeric_lib, wrapper_lib
end

function _find_paths(
    mode::CNPreferences.Developer,
    cupynumeric_jll_module::Module,
    cupynumeric_jll_wrapper_module::Nothing,
)
    cupynumeric_path = ""
    use_cupynumeric_jll = load_preference(CNPreferences, "cupynumeric_use_jll", true)

    if use_cupynumeric_jll == false
        cupynumeric_path = load_preference(CNPreferences, "cupynumeric_path", nothing)
        check_cupynumeric_install(cupynumeric_path)
    else
        check_jll(cupynumeric_jll_module)
        cupynumeric_path = cupynumeric_jll.artifact_dir
    end

    pkg_root = abspath(joinpath(@__DIR__, "../../"))
    wrapper_lib = joinpath(pkg_root, "lib", "cunumeric_jl_wrapper", "build", "lib")

    return joinpath(cupynumeric_path, "lib"), wrapper_lib
end

function _find_paths(
    mode::CNPreferences.Conda,
    cupynumeric_jll_module::Nothing,
    cupynumeric_jll_wrapper_module::Nothing,
)
    conda_env = load_preference(CNPreferences, "cunumeric_conda_env", nothing)
    isnothing(conda_env) && error(
        "cunumeric_conda_env preference must be set in LocalPreferences.toml when using conda mode"
    )

    check_cupynumeric_install(conda_env)
    pkg_root = abspath(joinpath(@__DIR__, "../../"))
    wrapper_lib = joinpath(pkg_root, "lib", "cunumeric_jl_wrapper", "build", "lib")

    return joinpath(conda_env, "lib"), wrapper_lib
end

# MPI, NCCL etc are found by Legate.find_dependency_paths
const DEPS_MAP = Dict(
    "CUTENSOR" => "libcutensor",
    "BLAS" => "libopenblas",
    "TBLIS" => "libtblis",
)
function find_dependency_paths(::Type{CNPreferences.JLL})
    results = Dict{String,String}()

    paths_to_search = copy(cupynumeric_jll.LIBPATH_list)

    for (name, lib) in DEPS_MAP
        results[name] = dirname(Libdl.find_library(lib, paths_to_search))
    end
    return results
end

find_dependency_paths(::Type{CNPreferences.Developer}) = Dict{String,String}()
find_dependency_paths(::Type{CNPreferences.Conda}) = Dict{String,String}()
