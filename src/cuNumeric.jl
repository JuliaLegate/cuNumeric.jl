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

include("utilities/depends.jl")
include("utilities/wrapper_download.jl")

const SUPPORTED_CUPYNUMERIC_VERSIONS = ["25.05.00"]

const DEFAULT_FLOAT = Float32
const DEFAULT_INT = Int32

const SUPPORTED_INT_TYPES = Union{Int32,Int64}
const SUPPORTED_FLOAT_TYPES = Union{Float32,Float64}
const SUPPORTED_NUMERIC_TYPES = Union{SUPPORTED_INT_TYPES,SUPPORTED_FLOAT_TYPES}
const SUPPORTED_TYPES = Union{SUPPORTED_INT_TYPES,SUPPORTED_FLOAT_TYPES,Bool} #* TODO Test UInt, Complex

# const MAX_DIM = 6 # idk what we compiled?

function preload_libs()
    libs = [
        joinpath(OpenBLAS32_jll.artifact_dir, "lib", "libopenblas.so"), # required for libcupynumeric.so
        joinpath(TBLIS_LIB, "libtblis.so"),
        joinpath(CUPYNUMERIC_LIB, "libcupynumeric.so"),
    ]

    if CUDA.functional()
        append!(libs, joinpath(CUTENSOR_LIB, "libcutensor.so"))
    end

    for lib in libs
        Libdl.dlopen(lib, Libdl.RTLD_GLOBAL | Libdl.RTLD_NOW)
    end
end

include("utilities/preference.jl")
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

# custom GC
include("memory.jl")

# allowscalar and allowpromotion
include("warnings.jl")

# NDArray internal
include("ndarray/detail/ndarray.jl")

# NDArray interface
include("ndarray/promotion.jl")
include("ndarray/broadcast.jl")
include("ndarray/ndarray.jl")
include("ndarray/unary.jl")
include("ndarray/binary.jl")

# scoping macro
include("scoping.jl")

# # Custom CUDA.jl kernel integration
if CUDA.functional()
    include("cuda.jl")
end

# # Utilities 
include("utilities/version.jl")
include("util.jl")

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
    if CUDA.functional()
        # in /src/cuda.jl to notify /wrapper/src/cuda.cpp about CUDA.jl kernel state size
        cuNumeric.set_kernel_state_size();
        # in /wrapper/src/cuda.cpp
        cuNumeric.register_tasks();
    end
    # setup /src/memory.jl 
    cuNumeric.init_gc!()
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

end #module cuNumeric
