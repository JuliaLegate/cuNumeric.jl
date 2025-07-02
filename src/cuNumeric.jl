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

using Legate
using CxxWrap
using Pkg
using Libdl

using LinearAlgebra
import LinearAlgebra: mul!

using Random
import Random: rand, rand!

import Base: abs, angle, acos, acosh, asin, asinh, atan, atanh, cbrt,
    ceil, clamp, conj, cos, cosh, cosh, deg2rad, exp, exp2, expm1,
    floor, frexp, imag, isfinite, isinf, isnan, log, log10,
    log1p, log2, !, modf, -, rad2deg, sign, signbit, sin,
    sinh, sqrt, tan, tanh, trunc, +, *, atan, &, |, ⊻, copysign,
    /, ==, ^, div, gcd, >, >=, hypot, isapprox, lcm, ldexp, <<,
    <, <=, !=, >>, all, any, argmax, argmin, maximum, minimum,
    prod, sum

function preload_libs()
    cache_build_meta = joinpath(@__DIR__, "../", "deps", "deps.jl")
    include(cache_build_meta)
    libs = [
        joinpath(CUTENSOR_ROOT, "lib", "libcutensor.so.2"),
        joinpath(HDF5_ROOT, "lib", "libhdf5.so.310"),
        joinpath(NCCL_ROOT, "lib", "libnccl.so.2"),
        joinpath(TBLIS_ROOT, "lib", "libtblis.so.0"),
    ]
    for lib in libs
        # @info "Preloading $lib"
        Libdl.dlopen(lib, Libdl.RTLD_GLOBAL | Libdl.RTLD_NOW)
    end
end

preload_libs()
lib = "libcupynumericwrapper.so"
libpath = joinpath(@__DIR__, "../", "wrapper", "build", lib)
@wrapmodule(() -> libpath)

include("capi.jl") # must go first
include("util.jl")
include("ndarray.jl")
include("unary.jl")
include("binary.jl")

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
    @info "Cleaning Up cuNuermic"
end

function cupynumeric_setup(AA::ArgcArgv)

    # Capture stdout from start_legate to 
    # see the hardware configuration

    # TODO CATCH STDERR
    # run(`bash -c "export LEGATE_AUTO_CONFIG=0"`)
    # run(`bash -c "export LEGATE_SHOW_CONFIG=1"`)
    # run(`bash -c "export LEGATE_CONFIG=\"--logging 2\""`)
    #println(ENV["LEGATE_AUTO_CONFIG"])
    #@info "LEGATE_AUTO_CONFIG: $(ENV["LEGATE_AUTO_CONFIG"])"
    #println(Base.get_bool_env("LEGATE_AUTO_CONFIG"))

    # cuNumeric.start_legate()
    #pipe = Pipe()
    #started = Base.Event()
    #writer = Threads.@spawn redirect_stdout(pipe) do
    #notify(started)
    #cuNumeric.start_legate()
    #close(Base.pipe_writer(pipe))
    #end

    #wait(started)
    #legate_config_str = Base.read(pipe, String)
    #wait(writer) 
    #print(legate_config_str)
    cuNumeric_config_str = ""

    @info "Started cuNuermic"

    Base.atexit(my_on_exit)

    cuNumeric.initialize_cunumeric(AA.argc, getargv(AA))

    return cuNumeric_config_str
end

global cuNumeric_config_str::String = ""

function versioninfo()
    msg = """
        CuNumeric Configuration: $(cuNumeric_config_str)
    """
    println(msg)
end

# Runtime initilization
# Called once in lifetime of code
function __init__()
    preload_libs()

    @initcxx
    # Legate ignores these arguments...
    AA = ArgcArgv([Base.julia_cmd()[1]])
    global cuNumeric_config = cupynumeric_setup(AA)
end
end
