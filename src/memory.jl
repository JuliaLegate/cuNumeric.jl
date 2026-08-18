using Base.Threads: Atomic, atomic_add!, atomic_sub!, atomic_xchg!, SpinLock

# Legate only permits handle destruction on the launch thread, but GC finalizers can
# run on another thread (e.g. 1.12's interactive thread), so they enqueue here and the
# launch thread drains later. Presized buffers keep the finalizer enqueue alloc-free.
const _RUNTIME_TID = Ref{Int}(0)
const _free_lock = SpinLock()
const _free_queue = Ptr{Cvoid}[]   # producers (finalizers) push
const _free_drain = Ptr{Cvoid}[]   # launch thread swaps into here to free

function _init_deferred_free!()
    _RUNTIME_TID[] = Threads.threadid()
    sizehint!(_free_queue, 1 << 16)
    sizehint!(_free_drain, 1 << 16)
    return nothing
end

# Runs in finalizers on any thread: no Legate call, no block, no steady-state alloc.
@inline function _enqueue_free!(ptr::Ptr{Cvoid})
    ptr == C_NULL && return nothing
    lock(_free_lock)
    push!(_free_queue, ptr)
    unlock(_free_lock)
    return nothing
end

@doc"""
    drain_pending_frees!()

Destroy NDArray handles queued by finalizers. No-op off the launch thread. Called
automatically from the op/allocation path, so user code rarely needs it.
"""
function drain_pending_frees!()
    Threads.threadid() == _RUNTIME_TID[] || return nothing
    isempty(_free_queue) && return nothing

    lock(_free_lock)
    append!(_free_drain, _free_queue)
    empty!(_free_queue)
    unlock(_free_lock)

    for ptr in _free_drain
        nda_destroy_array(ptr)
    end
    empty!(_free_drain)
    return nothing
end

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
# memory measured right after the last GC
const post_gc_device_bytes = Atomic{Int64}(0)
const post_gc_host_bytes = Atomic{Int64}(0)
# how much new memory must accumulate before GC fires again
const gc_hysteresis_frac = Ref{Float64}(0.05)

function get_number_of_runtimes()
    n = ccall((:nda_get_number_of_runtimes, libnda),
        Int32, ())
    return n
end

@doc"""
    init_gc!()

Initializes the cuNumeric garbage collector by querying the available
device memory and enabling the automatic GC heuristics.
"""
function init_gc!()
    total_device_bytes[] = query_total_device_memory()
    total_host_bytes[] = query_total_host_memory()
    # @info "[cuNumeric GC] $(total_device_bytes[]) framebuffer available"
    return AUTO_GC_ENABLE[] = true
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

    # minimum growth above the post-GC floor needed to re-collect
    dev_floor = post_gc_device_bytes[]
    host_floor = post_gc_host_bytes[]
    dev_delta = Int(round(gc_hysteresis_frac[] * total_device_bytes[]))
    host_delta = Int(round(gc_hysteresis_frac[] * total_host_bytes[]))
    grew = device_bytes > dev_floor + dev_delta || host_bytes > host_floor + host_delta

    if device_bytes > hard_limit(; host=false) || host_bytes > hard_limit()
        grew && _collect!(true)
    elseif device_bytes > soft_limit(; host=false) || host_bytes > soft_limit()
        grew && _collect!(false)
    else
        # reset floors so the next spike is caught immediately
        atomic_xchg!(post_gc_device_bytes, 0)
        atomic_xchg!(post_gc_host_bytes, 0)
    end

    return nothing
end

function _collect!(full::Bool)
    GC.gc(full)
    drain_pending_frees!()   # free what GC just enqueued
    recalibrate_allocator!()
    atomic_xchg!(post_gc_device_bytes, current_device_bytes[])
    atomic_xchg!(post_gc_host_bytes, current_host_bytes[])
    return nothing
end
