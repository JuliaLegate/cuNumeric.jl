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

using CUDA
using Legate
using CxxWrap
using Pkg
using Libdl
using TOML
import ExpressionExplorer: compute_symbols_state
import MacroTools: postwalk

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
    prod, sum, read

function preload_libs()
    cache_build_meta = joinpath(@__DIR__, "../", "deps", "deps.jl")
    include(cache_build_meta)
    libs = [
        joinpath(CUTENSOR_LIB, "libcutensor.so"),
        joinpath(TBLIS_LIB, "libtblis.so"),
    ]
    for lib in libs
        Libdl.dlopen(lib, Libdl.RTLD_GLOBAL | Libdl.RTLD_NOW)
    end
end

preload_libs()
lib = "libcupynumericwrapper.so"
libpath = joinpath(@__DIR__, "../", "wrapper", "build", lib)
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
    @info "Cleaning Up cuNuermic"
end

global cuNumeric_config_str::String = ""

function cunumeric_setup(AA::ArgcArgv)
    @info "Started cuNuermic"
    Base.atexit(my_on_exit)

    cuNumeric.initialize_cunumeric(AA.argc, getargv(AA))
    cuNumeric.register_tasks(); # in cuda.cpp wrapper interface
    cuNumeric.init_gc!() # setup memory.jl 
end

function versioninfo()
    println(cuNumeric_config_str)
end

# Runtime initilization
function __init__()
    preload_libs()
    @initcxx
    AA = ArgcArgv([Base.julia_cmd()[1]])
    global cuNumeric_config_str = version_config_setup()
    cunumeric_setup(AA)
end
end
