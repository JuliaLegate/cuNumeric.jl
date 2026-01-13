using Base.Threads: Atomic, atomic_add!, atomic_sub!, atomic_xchg!

query_device_memory() = ccall((:nda_query_device_memory, libnda),
    Int64, ())

const current_bytes = Atomic{Int64}(0)   # live, accounted allocations
# const pending_bytes = Atomic{Int}(0)   # predicted upcoming need
const total_bytes = Ref{Int64}(0)      # cached device total
const soft_frac = Ref{Float64}(0.80)
const hard_frac = Ref{Float64}(0.90)
const AUTO_GC_ENABLE = Ref{Bool}(false)

@doc"""
    init_gc!()

Initializes the cuNumeric garbage collector by querying the available
device memory and enabling the automatic GC heuristics.
"""
function init_gc!()
    total_bytes[] = query_device_memory()
    # @info "[cuNumeric GC] $(total_bytes[]) framebuffer available"
    AUTO_GC_ENABLE[] = true
end

@doc"""
    disable_gc!()

Disables the automatic garbage collection heuristics.
This gives the user full control over memory management.
"""
function disable_gc!(; verbose=true)
    AUTO_GC_ENABLE[] = false
    if verbose
        @info "You have disabled our GC heuristics. Good Luck!"
    end
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

function recalibrate_allocator!()
    # int nda_recalibrate_allocator(void);
    recal = ccall((:nda_recalibrate_allocator, libnda), Int64, (Cvoid,))
    @assert recal >= 0
    @info "[cuNumeric GC] Recalibrated allocator: $recal bytes"
    @info "[cuNumeric GC] Previous allocation: $(current_bytes[]) bytes"

    atomic_xchg!(current_bytes, recal)
    return nothing
end

function maybe_collect()
    # cur = current_bytes[]
    # pend = pending_bytes[]
    # tot = cur + pend

    tot = current_bytes[]
    if tot > hard_limit()
        # Aggressive
        GC.gc(true)
        recalibrate_allocator!()
    elseif tot > soft_limit()
        # Gentle
        GC.gc(false)
        recalibrate_allocator!()
    end
    return nothing
end
