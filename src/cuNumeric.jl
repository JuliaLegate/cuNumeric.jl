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
 *            Nader Rahhal <naderrahhal2026@u.northwestern.edu>
=#

module cuNumeric

using Preferences
using CNPreferences
using LegatePreferences: LegatePreferences
using Legate
using Libdl
using CxxWrap

using CUDATools: CUDATools
using CUDACore: CUDACore
import CUDACore: CuArray
import KernelAbstractions: @kernel, @index
import KernelAbstractions as KA

using cupynumeric_jll
using cunumeric_jl_wrapper_jll

import Base: axes, convert, copy, copyto!, inv, isfinite, sqrt, -, +, *, ==, !=,
    isapprox, read, view, maximum, minimum, prod, sum, getindex, setindex!,
    sum, prod, argmax, argmin

using LinearAlgebra
import LinearAlgebra: mul!

import AbstractFFTs: fft, ifft, fft!, ifft!

using Random
import Random: rand!, randn!, randexp!

using StaticArrays: SVector

using StatsBase
import StatsBase: var, mean, std

include(joinpath(@__DIR__, "../deps/version.jl"))
include("utilities/preference.jl")

const HAS_CUDA = LegatePreferences.has_cuda_gpu()
if !HAS_CUDA
    @warn "We couldn't find a CUDA-enabled GPU. If you have an NVIDIA GPU something might be wrong."
end

# `HAS_CUDA` describes the machine. A CPU-only Legate configuration on a GPU
# machine must still avoid registering or launching GPU tasks.
@inline _has_gpu_target() = HAS_CUDA && Int(Legate.num_gpus()) > 0

const DEFAULT_FLOAT = Float32
const DEFAULT_INT = Int32

const SUPPORTED_INT_TYPES = Union{Int8,Int16,Int32,Int64,UInt8,UInt16,UInt32,UInt64}
const SUPPORTED_FLOAT_TYPES = Union{Float32,Float64} # Float16 disabled for now. Issues need to be resolved.
const SUPPORTED_COMPLEX_TYPES = Union{ComplexF32,ComplexF64}

const SUPPORTED_NUMERIC_TYPES = Union{
    SUPPORTED_INT_TYPES,SUPPORTED_FLOAT_TYPES,SUPPORTED_COMPLEX_TYPES
}

# solve has no integer backend kernel
const SUPPORTED_SOLVE_TYPES = Union{SUPPORTED_FLOAT_TYPES,SUPPORTED_COMPLEX_TYPES}
const SUPPORTED_SVD_TYPES = Union{SUPPORTED_FLOAT_TYPES,SUPPORTED_COMPLEX_TYPES}
const SUPPORTED_QR_TYPES = Union{SUPPORTED_FLOAT_TYPES,SUPPORTED_COMPLEX_TYPES}
const SUPPORTED_CHOLESKY_TYPES = Union{SUPPORTED_FLOAT_TYPES,SUPPORTED_COMPLEX_TYPES}
const SUPPORTED_EIG_TYPES = Union{SUPPORTED_FLOAT_TYPES,SUPPORTED_COMPLEX_TYPES}
const SUPPORTED_ARRAY_TYPES = Union{Bool,SUPPORTED_NUMERIC_TYPES}
const SUPPORTED_TYPES = Union{SUPPORTED_ARRAY_TYPES,String}

# const MAX_DIM = 6 # idk what we compiled?

# Sets the LEGATE_LIB_PATH and WRAPPER_LIB_PATH preferences based on mode
# This will also include the relevant JLLs if necessary.
MODE = load_preference(CNPreferences, "cunumeric_mode", CNPreferences.MODE_JLL)
@static if MODE == CNPreferences.MODE_JLL
    using cupynumeric_jll, cunumeric_jl_wrapper_jll
    find_paths(
        MODE;
        cupynumeric_jll_module=cupynumeric_jll,
        cupynumeric_jll_wrapper_module=cunumeric_jl_wrapper_jll,
    )
elseif MODE == CNPreferences.MODE_DEVELOPER
    use_cupynumeric_jll = load_preference(CNPreferences, "legate_use_jll", true)
    if use_cupynumeric_jll
        using cupynumeric_jll
        find_paths(
            MODE;
            cupynumeric_jll_module=cupynumeric_jll,
            cupynumeric_jll_wrapper_module=nothing,
        )
    else
        find_paths(MODE)
    end
elseif MODE == CNPreferences.MODE_CONDA
    find_paths(MODE)
else
    error(
        "cuNumeric.jl: Unknown mode $(MODE). Must be one of 'jll', 'developer', or 'conda'."
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

# Compile-time so task scope instrumentation is fully elided when disabled.
const TASK_SCOPE_NAMES = CNPreferences.TASK_SCOPE_NAMES

# NDArray internal
include("ndarray/detail/ndarray.jl")
include("ndarray/detail/linalg.jl")
include("ndarray/detail/fft.jl")

# Utilities
include("cuda/strided_device_array.jl")
include("cuda/cuda_util.jl")
include("utilities/version.jl")
include("util.jl")
include("mpi_bootstrap.jl")

# Compile-time so the fusion branch is elided; flip via CNPreferences before loading.
const FUSE_BROADCAST_EXPRS = CNPreferences.FUSE_BROADCAST
# Fuse when broadcast tree length is at least this (see `_broadcast_tree_length`).
# Default 2 skips single-op exprs like `y .= cos.(x)`. Use 1 to fuse everything.
const FUSE_BROADCAST_MIN_OPS = CNPreferences.FUSE_BROADCAST_MIN_OPS

# Functionality
include("ndarray/diagonal.jl")
include("ndarray/promotion.jl")
include("cuda/cuda_ptx_task.jl")
include("ndarray/broadcast_fusion.jl")
include("ndarray/broadcast.jl")
include("ndarray/ndarray.jl")
include("ndarray/random/bitgenerator.jl")
include("ndarray/random/generator.jl")
include("ndarray/random/random.jl")
include("ndarray/unary.jl")
include("ndarray/binary.jl")
include("ndarray/linalg.jl")
include("ndarray/batched_linalg.jl")
include("ndarray/contract.jl")
include("ndarray/fft.jl")
include("scoping/scoping.jl")

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
    return drain_pending_frees!()   # flush before Legate tears down
end

global cuNumeric_config_str::String = ""

### These functions guard against a user trying
### to start multiple runtimes and also to allow
## package extensions which always try to re-load

const RUNTIME_INACTIVE = -1
const RUNTIME_ACTIVE = 0
const _runtime_ref = Ref{Int}(RUNTIME_INACTIVE)
const _start_lock = ReentrantLock()

runtime_started() = _runtime_ref[] == RUNTIME_ACTIVE

function _start_runtime()
    Libdl.dlopen(CUPYNUMERIC_LIB_PATH, Libdl.RTLD_GLOBAL | Libdl.RTLD_NOW)
    Libdl.dlopen(CUPYNUMERIC_WRAPPER_LIB_PATH, Libdl.RTLD_GLOBAL | Libdl.RTLD_NOW)

    AA = ArgcArgv(String[])
    # AA = ArgcArgv([Base.julia_cmd()[1]])
    cuNumeric.initialize_cunumeric(AA.argc, getargv(AA))

    _init_deferred_free!()   # record launch thread for deferred frees (memory.jl)

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

_is_precompiling() = ccall(:jl_generating_output, Cint, ()) != 0

# Distributed stdlib UUID; looked up without a hard dependency.
const _DISTRIBUTED_PKGID = Base.PkgId(
    Base.UUID("8ba89e20-285c-5b6f-9357-94700520ee1b"), "Distributed"
)

# A worker only reaches here correctly if setup_legate_env set the sentinel + P2P env
# before `using cuNumeric`. Anything else (julia -p N, bare Distributed.addprocs) would
# start a misconfigured runtime, so refuse it.
function _assert_worker_configured()
    dist = get(Base.loaded_modules, _DISTRIBUTED_PKGID, nothing)
    dist === nothing && return nothing                            # not a Distributed session
    dist.myid() == 1 && return nothing                            # driver, not a worker
    haskey(ENV, "CUNUMERIC_DISTRIBUTED_WORKER") && return nothing # set by our setup path
    return error(
        "cuNumeric loaded on an unconfigured Distributed worker (id $(dist.myid())). " *
        "Start workers with `cuNumeric.addprocs(...)`, or `cuNumeric.init_workers()` if " *
        "you launched them yourself; `julia -p N` and bare `Distributed.addprocs` are " *
        "unsupported.",
    )
end

# Runtime initilization
function __init__()
    CNPreferences.check_unchanged()

    @initcxx

    _is_precompiling() && return nothing

    # Registry CI machines can't set LEGATE_CONFIG, so don't start the runtime there.
    get(ENV, "JULIA_REGISTRYCI_AUTOMERGE", false) == "true" && return nothing

    # Choose the networking bootstrap. Under an MPI launcher every rank is a Legate rank
    # (SPMD); otherwise validate the Distributed/p2p worker before it can start a runtime.
    mpi = _detect_mpi_bootstrap()
    mpi === nothing ? _assert_worker_configured() : _configure_mpi_bootstrap!(mpi)

    get(ENV, "LEGATE_SKIP_RUNTIME", false) == "true" && return nothing

    # Start runtime, but only if not pre-compiling
    ensure_runtime!()

    # Requries runtime to be started
    return _setup_cuda_tasking()
end

end #module cuNumeric
