module DistributedExt

using Distributed
using Sockets
# Don't import cuNumeric here - it would initialize runtime before env vars are set!
include("setup.jl")
include("ports.jl")

function __init__()
    # Get cuNumeric module from loaded modules without importing it
    cunumeric_pkgid = Base.PkgId(Base.UUID("0fd9ffd4-7e84-4cd0-b8f8-645bd8c73620"), "cuNumeric")
    cuNumeric = Base.loaded_modules[cunumeric_pkgid]
    # Register the init_workers function at runtime
    Base.@eval cuNumeric init_workers(; kwargs...) = $DistributedExt.init_workers_impl(; kwargs...)
end

end # module DistributedExt
