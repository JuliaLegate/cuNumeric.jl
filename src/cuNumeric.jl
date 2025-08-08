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

module cuNumeric

include("depends.jl")

const SUPPORTED_CUPYNUMERIC_VERSIONS = ["25.05.00"]

function preload_libs()
    libs = [
        joinpath(OpenBLAS32_jll.artifact_dir, "lib", "libopenblas.so"), # required for libcupynumeric.so
        joinpath(CUTENSOR_LIB, "libcutensor.so"),
        joinpath(TBLIS_LIB, "libtblis.so"),
        joinpath(CUPYNUMERIC_LIB, "libcupynumeric.so"),
    ]
    for lib in libs
        Libdl.dlopen(lib, Libdl.RTLD_GLOBAL | Libdl.RTLD_NOW)
    end
end

include("preference.jl")
find_preferences()

const BLAS_LIB = load_preference(CNPreferences, "BLAS_LIB", nothing)
const CUTENSOR_LIB = load_preference(CNPreferences, "CUTENSOR_LIB", nothing)
const TBLIS_LIB = load_preference(CNPreferences, "TBLIS_LIB", nothing)
const CUPYNUMERIC_LIB = load_preference(CNPreferences, "CUPYNUMERIC_LIB", nothing)
const CUNUMERIC_WRAPPER_LIB = load_preference(CNPreferences, "CUNUMERIC_WRAPPER_LIB", nothing)

libnda = joinpath(CUNUMERIC_WRAPPER_LIB, "libcunumeric_c_wrapper.so")
libpath = joinpath(CUNUMERIC_WRAPPER_LIB, "libcunumeric_jl_wrapper.so")
if !isfile(libpath)
    error("Developer mode: You need to call Pkg.build()")
end

preload_libs() # for precompilation

@wrapmodule(() -> libpath)

include("version.jl") # version_config_setup
include("memory.jl") # memory gc before c-array 
include("capi.jl") # c-array interface prior to ndarray
include("util.jl")
include("ndarray.jl")
include("unary.jl")
include("binary.jl")
include("cuda.jl")

# From https://github.com/JuliaGraphics/QML.jl/blob/dca239404135d85fe5d4afe34ed3dc5f61736c63/src/QML.jl#L147
mutable struct ArgcArgv
    argv
    argc::Cint

    function ArgcArgv(args::Vector{String})
        argv = Base.cconvert(CxxPtr{CxxPtr{CxxChar}}, args)
        argc = length(args)
        return new(argv, argc)
    end
end

getargv(a::ArgcArgv) = Base.unsafe_convert(CxxPtr{CxxPtr{CxxChar}}, a.argv)

function my_on_exit()
    # @info "Cleaning Up cuNuermic"
end

global cuNumeric_config_str::String = ""

function cunumeric_setup(AA::ArgcArgv)
    Base.atexit(my_on_exit)

    cuNumeric.initialize_cunumeric(AA.argc, getargv(AA))
    cuNumeric.register_tasks(); # in cuda.cpp wrapper interface
    cuNumeric.init_gc!() # setup memory.jl 
end

@doc"""
    versioninfo()

Prints the cuNumeric build configuration summary, including package
metadata, Julia and compiler version, and paths to core dependencies.
"""
function versioninfo()
    println(cuNumeric_config_str)
end

# Runtime initilization
function __init__()
    CNPreferences.check_unchanged()
    preload_libs()
    @initcxx

    AA = ArgcArgv([Base.julia_cmd()[1]])
    global cuNumeric_config_str = version_config_setup()
    cunumeric_setup(AA)
end
end
