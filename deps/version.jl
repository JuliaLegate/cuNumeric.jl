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

const MIN_CUDA_VERSION = v"13.0"
const MAX_CUDA_VERSION = v"13.9.999"
const MIN_CUNUMERIC_VERSION = v"26.01.00"
const MAX_CUNUMERIC_VERSION = v"26.12.00"

up_dir(dir::String) = abspath(joinpath(dir, ".."))

function get_version(version_file::String)
    version = nothing
    open(version_file, "r") do f
        data = readlines(f)
        major = parse(Int, split(data[end - 2])[end])
        minor = parse(Int, lpad(split(data[end - 1])[end], 2, '0'))
        patch = parse(Int, lpad(split(data[end])[end], 2, '0'))
        version = VersionNumber(major, minor, patch)
    end
    if isnothing(version)
        error("cuNumeric.jl: Failed to parse version for $(version_file)")
    end
    return version
end

function get_cupynumeric_version(cupynumeric_root::String)
    version_file = joinpath(cupynumeric_root, "include", "cupynumeric", "version_config.hpp")
    return get_version(version_file)
end

function is_supported_version(version::VersionNumber)
    return MIN_CUNUMERIC_VERSION <= version && version <= MAX_CUNUMERIC_VERSION
end

function cupynumeric_valid(cupynumeric_root::String)
    version_cupynumeric = get_cupynumeric_version(cupynumeric_root)
    return is_supported_version(version_cupynumeric)
end

function is_cupynumeric_installed(cupynumeric::String; throw_errors::Bool=false)
    include_dir = joinpath(cupynumeric, "include")
    if !isdir(joinpath(include_dir, "cupynumeric/cupynumeric"))
        throw_errors &&
            @error "cuNumeric.jl: Cannot find include/cupynumeric/cupynumeric in $(cupynumeric)"
        return false
    end
    return true
end
