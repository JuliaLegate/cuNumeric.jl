using Base.Threads: Atomic, atomic_add!, atomic_sub!, atomic_xchg!

query_total_device_memory() = ccall((:nda_query_total_device_memory, libnda),
    Int64, ())
query_total_host_memory() = ccall((:nda_query_total_host_memory, libnda),
    Int64, ())
nda_query_allocated_device_memory() = ccall((:nda_query_allocated_device_memory, libnda),
    Int64, ())
nda_query_allocated_host_memory() = ccall((:nda_query_allocated_host_memory, libnda),
    Int64, ())

const total_device_bytes = Ref{Int64}(0)      # cached device total
const total_host_bytes = Ref{Int64}(0)      # cached host total
const current_device_bytes = Atomic{Int64}(0)   # predicted device allocations
const current_host_bytes = Atomic{Int64}(0)   # predicted host allocations
const soft_frac = Ref{Float64}(0.80)
const hard_frac = Ref{Float64}(0.90)
const AUTO_GC_ENABLE = Ref{Bool}(false)

@doc"""
    init_gc!()

Initializes the cuNumeric garbage collector by querying the available
device memory and enabling the automatic GC heuristics.
"""
function init_gc!()
    total_device_bytes[] = query_total_device_memory()
    total_host_bytes[] = query_total_host_memory()
    # @info "[cuNumeric GC] $(total_device_bytes[]) framebuffer available"
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

@inline _limit(frac, host) = Int(round(frac[] * (host ? total_host_bytes[] : total_device_bytes[])))

soft_limit(; host=true) = _limit(soft_frac, host)
hard_limit(; host=true) = _limit(hard_frac, host)

function register_alloc!(nbytes::Integer)
    # assume device allocation if we have a GPU
    # the recalibration phase will fix any discrepancies
    if HAS_CUDA
        atomic_add!(current_device_bytes, nbytes)
    else
        atomic_add!(current_host_bytes, nbytes)
    end

    gc_flag = AUTO_GC_ENABLE[]
    if gc_flag == true
        maybe_collect()
    end
    return nothing
end

function register_free!(nbytes::Integer)
    if HAS_CUDA
        atomic_sub!(current_device_bytes, nbytes)
    else
        atomic_sub!(current_host_bytes, nbytes)
    end
    return nothing
end

function recalibrate_allocator!()
    recal_host_mem = ccall((:nda_query_allocated_host_memory, libnda), Int64, ())
    @assert recal_host_mem >= 0
    atomic_xchg!(current_host_bytes, recal_host_mem)

    if HAS_CUDA
        recal_device_mem = ccall((:nda_query_allocated_device_memory, libnda), Int64, ())
        @assert recal_device_mem >= 0
        atomic_xchg!(current_device_bytes, recal_device_mem)
    end

    return nothing
end

function maybe_collect()
    host_bytes = current_host_bytes[]
    device_bytes = current_device_bytes[]
    if host_bytes > hard_limit() || device_bytes > hard_limit(; host=false)
        # Aggressive
        GC.gc(true)
        recalibrate_allocator!()
    elseif host_bytes > soft_limit() || device_bytes > soft_limit(; host=false)
        # Gentle
        GC.gc(false)
        recalibrate_allocator!()
    end

    return nothing
end
