"""
    addprocs(n::Integer; kwargs...)

Convenience wrapper that combines `Distributed.addprocs()` and `cuNumeric.init_workers()`.
Starts workers and automatically configures them for distributed cuNumeric with p2p networking.

# Example
```julia
using Distributed
using cuNumeric

# Start and configure workers in one call
cuNumeric.addprocs(4)

# Ready to use distributed cuNumeric!
@everywhere workers() begin
    a = cuNumeric.rand(100)
end
```

All keyword arguments are passed through to `Distributed.addprocs()`.
"""
function addprocs_impl(n::Integer; kwargs...)
    # Start workers
    pids = Distributed.addprocs(n; kwargs...)

    # Configure them for cuNumeric
    init_workers_impl()

    return pids
end

function init_workers_impl(; auto_setup::Bool=true)
    w = Distributed.workers()
    if isempty(w)
        @warn "No Distributed.jl workers found. Did you call addprocs()?"
        return nothing
    end

    @info "Setting up cuNumeric on $(length(w)) workers..."

    if !auto_setup
        @info "✓ Skipping automatic p2p setup (auto_setup=false)"
        return nothing
    end

    # Get cuNumeric package directory to load port utilities
    cunumeric_pkgid = Base.PkgId(Base.UUID("0fd9ffd4-7e84-4cd0-b8f8-645bd8c73620"), "cuNumeric")
    cunumeric_path = dirname(dirname(Base.locate_package(cunumeric_pkgid)))
    port_path = joinpath(cunumeric_path, "ext", "DistributedExt", "ports.jl")

    # Use @everywhere with myid() check (not @everywhere workers() - that doesn't work!)
    Base.eval(Main, :(@everywhere begin
        if myid() != 1
            include($port_path)
            setup_legate_env()
            using cuNumeric
            @info "Number of runtimes: " cuNumeric.get_number_of_runtimes()
        end
    end))

    @info "✓ cuNumeric loaded on all workers with p2p networking"

    return nothing
end
