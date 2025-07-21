using Base.Threads: Atomic, atomic_add!, atomic_sub!, atomic_xchg!

# get_device_mem(dev::Integer=0) = (free=CUDA.memory_status(dev)... )  

const current_bytes = Atomic{Int}(0)   # live, accounted allocations
const pending_bytes = Atomic{Int}(0)   # predicted upcoming need
const total_bytes = Ref{Int}(0)      # cached device total
const soft_frac = Ref{Float64}(0.80)
const hard_frac = Ref{Float64}(0.90)

# Refresh total (call at init or when device changes)
function refresh_total!(dev::Integer=0)
    _, tot = get_device_mem(dev)
    total_bytes[] = tot
    return tot
end

soft_limit() = Int(round(soft_frac[] * total_bytes[]))
hard_limit() = Int(round(hard_frac[] * total_bytes[]))

function register_alloc!(nbytes::Integer)
    atomic_add!(current_bytes, nbytes)
    maybe_collect(:alloc, nbytes)
    return nothing
end

function register_free!(nbytes::Integer)
    atomic_sub!(current_bytes, nbytes)
    return nothing
end

"""
    maybe_collect(reason, newbytes)

Soft: `GC.gc(false)` (non-full); Hard: `GC.gc(true)`
"""
function maybe_collect(reason::Symbol, newbytes::Integer=0)
    cur = current_bytes[]
    pend = pending_bytes[]
    tot = cur + pend
    if tot > hard_limit()
        # Aggressive
        GC.gc(true)
        trim_gpu_pools!()
    elseif tot > soft_limit()
        # Gentle
        GC.gc(false)
    end
    return nothing
end
