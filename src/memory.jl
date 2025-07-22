using Base.Threads: Atomic, atomic_add!, atomic_sub!, atomic_xchg!

lib = "libcwrapper.so"
libnda = joinpath(@__DIR__, "../", "wrapper", "build", lib)

query_device_memory() = ccall((:nda_query_device_memory, libnda),
    Int64, ())

const current_bytes = Atomic{Int64}(0)   # live, accounted allocations
# const pending_bytes = Atomic{Int}(0)   # predicted upcoming need
const total_bytes = Ref{Int64}(0)      # cached device total
const soft_frac = Ref{Float64}(0.80)
const hard_frac = Ref{Float64}(0.90)
const AUTO_GC_ENABLE = Ref{Bool}(false)

function init_gc!()
    total_bytes[] = query_device_memory()
    # @info "[cuNumeric GC] $(total_bytes[]) framebuffer available"
    AUTO_GC_ENABLE[] = true
end

function disable_gc!()
    AUTO_GC_ENABLE[] = false
    @info "You have disabled our GC heuristics. Good Luck!"
end

soft_limit() = Int(round(soft_frac[] * total_bytes[]))
hard_limit() = Int(round(hard_frac[] * total_bytes[]))

function register_alloc!(nbytes::Integer)
    atomic_add!(current_bytes, nbytes)
    gc_flag = AUTO_GC_ENABLE[]
    if gc_flag == true
        maybe_collect()
    end
    return nothing
end

function register_free!(nbytes::Integer)
    atomic_sub!(current_bytes, nbytes)
    return nothing
end

"""
    maybe_collect()

Soft: `GC.gc(false)` (non-full); Hard: `GC.gc(true)`
"""
function maybe_collect()
    # cur = current_bytes[]
    # pend = pending_bytes[]
    # tot = cur + pend

    tot = current_bytes[]
    if tot > hard_limit()
        # Aggressive
        GC.gc(true)
    elseif tot > soft_limit()
        # Gentle
        GC.gc(false)
    end
    return nothing
end
