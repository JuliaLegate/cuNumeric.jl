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

const HAS_CUDA = cupynumeric_jll.host_platform["cuda"] != "none"

if !HAS_CUDA
    @warn "cuPyNumeric JLL does not have CUDA. If you have an NVIDIA GPU something might be wrong."
end

const SUPPORTED_CUPYNUMERIC_VERSIONS = ["25.10.00", "25.11.00"]

const DEFAULT_FLOAT = Float32
const DEFAULT_INT = Int32

const SUPPORTED_INT_TYPES = Union{Int32,Int64}
const SUPPORTED_FLOAT_TYPES = Union{Float32,Float64}
const SUPPORTED_NUMERIC_TYPES = Union{SUPPORTED_INT_TYPES,SUPPORTED_FLOAT_TYPES}
const SUPPORTED_TYPES = Union{SUPPORTED_INT_TYPES,SUPPORTED_FLOAT_TYPES,Bool} #* TODO Test UInt, Complex

# const MAX_DIM = 6 # idk what we compiled?

include("utilities/preference.jl")

# Sets the LEGATE_LIB_PATH and WRAPPER_LIB_PATH preferences based on mode
# This will also include the relevant JLLs if necessary.
@static if CNPreferences.MODE == "jll"
    using cupynumeric_jll, cunumeric_jl_wrapper_jll
    find_paths(
        CNPreferences.MODE;
        cupynumeric_jll_module=cupynumeric_jll,
        cupynumeric_jll_wrapper_module=cunumeric_jl_wrapper_jll,
    )
elseif CNPreferences.MODE == "developer"
    use_cupynumeric_jll = load_preference(CNPreferences, "legate_use_jll", true)
    if use_cupynumeric_jll
        using cupynumeric_jll
        find_paths(
            CNPreferences.MODE;
            cupynumeric_jll_module=cupynumeric_jll,
            cupynumeric_jll_wrapper_module=nothing,
        )
    else
        find_paths(CNPreferences.MODE)
    end
elseif CNPreferences.MODE == "conda"
    using cunumeric_jl_wrapper_jll
    find_paths(
        CNPreferences.MODE,
        cupynumeric_jll_module=nothing,
        cupynumeric_jll_wrapper_module=cunumeric_jl_wrapper_jll,
    )
else
    error(
        "cuNumeric.jl: Unknown mode $(CNPreferences.MODE). Must be one of 'jll', 'developer', or 'conda'."
    )
end

const CUPYNUMERIC_LIBDIR = load_preference(CNPreferences, "CUPYNUMERIC_LIBDIR", nothing)
const CUPYNUMERIC_WRAPPER_LIBDIR = load_preference(
    CNPreferences, "CUPYNUMERIC_WRAPPER_LIBDIR", nothing
)

const libnda = joinpath(CUPYNUMERIC_WRAPPER_LIBDIR, "libcunumeric_c_wrapper.so")
const CUPYNUMERIC_WRAPPER_LIB_PATH = joinpath(
    CUPYNUMERIC_WRAPPER_LIBDIR, "libcunumeric_jl_wrapper.so"
)
const CUPYNUMERIC_LIB_PATH = joinpath(CUPYNUMERIC_LIBDIR, "libcupynumeric.so")

(isnothing(CUPYNUMERIC_LIBDIR) || isnothing(CUPYNUMERIC_WRAPPER_LIBDIR)) && error(
    "cuNumeric.jl: CUPYNUMERIC_LIBDIR or CUPYNUMERIC_WRAPPER_LIBDIR preference not set. Check LocalPreferences.toml"
)

if !isfile(CUPYNUMERIC_WRAPPER_LIB_PATH)
    # Print build error logs if available
    deps = joinpath(dirname(@__DIR__), "deps")
    for errfile in ["cpp_wrapper.err", "libcxxwrap.err"]
        errpath = joinpath(deps, errfile)
        if isfile(errpath)
            println("\n=== Contents of $errfile ===")
            println(read(errpath, String))
            println("=== End of $errfile ===\n")
        end
    end
    error(
        "Developer mode: You need to call Pkg.build(). Library $CUPYNUMERIC_WRAPPER_LIB_PATH not found."
    )
end

@wrapmodule(() -> CUPYNUMERIC_WRAPPER_LIB_PATH)

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

# Utilities 
include("utilities/version.jl")
include("utilities/cuda_stubs.jl")
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
    # @info "Cleaning Up cuNumeric"
end

global cuNumeric_config_str::String = ""

@doc"""
    versioninfo()

Prints the cuNumeric build configuration summary, including package
metadata, Julia and compiler version, and paths to core dependencies.
"""
function versioninfo()
    println(cuNumeric_config_str)
end

function setup_runtime(; pidfile = "/pool/emeitz/test.pid")
    res = trymkpidlock(pidfile) do 
        # AA = ArgcArgv(String[])
        # AA = ArgcArgv([Base.julia_cmd()[1]])
        # cuNumeric.initialize_cunumeric(AA.argc, getargv(AA))
    end

    if res !== nothing
        @warn "Could not acquire cuNumeric runtime lock. Another Julia process may be using cuNumeric or Legate."
    end

    return nothing
end

### These functions guard against a user trying
### to start multiple runtimes and also to allow
## package extensions which always try to re-load

const RUNTIME_INACTIVE = -1
const RUNTIME_ACTIVE = 0
const _runtime_ref = Ref{Int}(RUNTIME_INACTIVE)
const _start_lock  = ReentrantLock()

runtime_started() = _runtime_ref[] == RUNTIME_ACTIVE

function _start_runtime()

    CNPreferences.check_unchanged()

    Libdl.dlopen(CUPYNUMERIC_LIB_PATH, Libdl.RTLD_GLOBAL | Libdl.RTLD_NOW)
    Libdl.dlopen(CUPYNUMERIC_WRAPPER_LIB_PATH, Libdl.RTLD_GLOBAL | Libdl.RTLD_NOW)

    @initcxx

    global cuNumeric_config_str = version_config_setup()

    AA = ArgcArgv(String[])
    # AA = ArgcArgv([Base.julia_cmd()[1]])
    cuNumeric.initialize_cunumeric(AA.argc, getargv(AA))

    # setup /src/memory.jl 
    cuNumeric.init_gc!()

    Base.atexit(my_on_exit)


    return RUNTIME_ACTIVE
end

function ensure_runtime!()
    # fast path (no lock)
    rt = _runtime_ref[]
    (rt == RUNTIME_INACTIVE) || return rt

    lock(_start_lock)
    try
        # re-check after lock
        rt = _runtime_ref[]
        (rt == RUNTIME_INACTIVE) || return rt

        rt = _start_runtime()
        _runtime_ref[] = rt
        return rt
    finally
        unlock(_start_lock)
    end
end

# Runtime initilization
function __init__()
    @info "cuNumeric __init__" pid=getpid() tid=Threads.threadid()

    ensure_runtime!()
end

end #module cuNumeric
