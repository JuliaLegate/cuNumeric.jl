export @cunumeric

using MacroTools
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

macro __tryfinally(ex, fin)
    Expr(:tryfinally,
        :($(esc(ex))),
        :($(esc(fin))),
    )
end

macro cunumeric(block)
    esc(process_ndarray_scope(block))
end

const ndarray_scope_cache = Dict{UInt64,Expr}()
const counter = Ref(0)

function process_ndarray_scope(block)
    # Normalize block to list of statements
    stmts = block isa Expr && block.head == :block ? block.args : [block]

    # Hash the content structurally (avoiding pointer identity)
    h = hash(stmts)

    # Return cached result if present
    if haskey(ndarray_scope_cache, h)
        return ndarray_scope_cache[h]
    end

    # Otherwise, process and cache
    assigned_vars = Set{Symbol}()
    body = Any[]

    for stmt in stmts
        stmt = find_ndarray_assignments(stmt, assigned_vars)
        push!(body, stmt)
    end

    cleanup_exprs = []

    for var in assigned_vars
        push!(cleanup_exprs, :(cuNumeric.nda_destroy_array($var.ptr)))
        push!(cleanup_exprs, :(cuNumeric.register_free!($var.nbytes)))
        push!(cleanup_exprs, :($var.ptr = Ptr{Cvoid}(0)))
        push!(cleanup_exprs, :($var.nbytes = 0))
    end

    result = quote
        cuNumeric.@__tryfinally($(Expr(:block, body...)),
            $(Expr(:block, cleanup_exprs...)))
    end

    counter[] = 0
    ndarray_scope_cache[h] = result
    return result
end

function find_ndarray_assignments(ex, assigned_vars::Set{Symbol})
    cache = Dict{Expr,Symbol}()
    function rewrite(e)::Tuple{Any,Vector{Expr}}
        if !(e isa Expr)
            return e, Expr[]
        end

        if e.head == :(=)
            lhs, rhs = e.args
            if lhs isa Symbol
                push!(assigned_vars, lhs)
            end
            new_rhs, temps = rewrite(rhs)
            return Expr(:block, temps..., :($lhs = $new_rhs)), []
        elseif e.head == :ref
            if haskey(cache, e)
                return cache[e], []
            else
                counter[] += 1
                tmp = Symbol(:tmp, counter[]) # make tmp1, tmp2, tmp3 ....
                cache[e] = tmp
                push!(assigned_vars, tmp)
                return tmp, [:($tmp = $e)]
            end
        else
            new_args = Any[]
            hoisted = Expr[]
            for arg in e.args
                new_arg, temps = rewrite(arg)
                push!(new_args, new_arg)
                append!(hoisted, temps)
            end
            return Expr(e.head, new_args...), hoisted
        end
    end

    new_ex, temps = rewrite(ex)
    return Expr(:block, temps..., new_ex)
end
