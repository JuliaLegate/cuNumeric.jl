export @analyze_lifetimes, @show_lifetimes

# Include generic syntax layers before the cuNumeric-specific lifetime passes.
include("util.jl")
using .ScopingUtils

include("inter_broadcast_fusion.jl")

@doc"""
    @analyze_lifetimes expr

Wraps a block of code so that all temporary `NDArray` allocations
(e.g. from slicing or function calls) are tracked and safely freed
at the end of the block. Ensures proper cleanup of GPU memory by
inserting `maybe_insert_delete` calls automatically.

When broadcast fusion is enabled (`FUSE_BROADCAST_EXPRS`), dotted operators
(`.+`, `.*`, etc.) form a lazy `Base.Broadcast.Broadcasted` tree compiled into
a single PTX kernel; intermediate nodes are not real `NDArray` allocations and
are not individually hoisted. The macro automatically selects the
broadcast-aware analysis in that case and the plain analysis otherwise.
"""
macro analyze_lifetimes(block)
    on_rewrite = BCAST_FUSION_DEBUG[] ? InterBroadcastFusion.log_rewrite : nothing
    return esc(process_ndarray_scope(block; on_rewrite))
end

const counter = Ref(0)

function maybe_insert_delete(var::NDArray)
    return cuNumeric.destroy!(var)
end

maybe_insert_delete(x) = x

function _hoist_temporary(expr, assigned_vars)
    counter[] += 1
    temporary = Symbol(:tmp, counter[])
    push!(assigned_vars, temporary)
    return temporary, [:($temporary = $expr)]
end

# Turn the named allocations produced by either lifetime rewriter into an
# executable scope. For example, if tmp1 and tmp2 are last used by statement 3:
#
#   3  tmp3 = f(tmp1, tmp2)
#      maybe_insert_delete(tmp1)
#      maybe_insert_delete(tmp2)
#
# The final value of the block is protected because it escapes to the caller.
"""
    insert_finalizers(stmts::Vector)
Insert `cuNumeric.maybe_insert_delete(var)` after the last use of each temporary variable.
"""
function insert_finalizers(exprs::Vector, assigned_vars::Set{Symbol})
    last_use = Dict{Symbol,Int}()
    alias_map = Dict{Symbol,Symbol}()

    stmts = _flatten_statements(Expr(:block, exprs...))

    # Pass 1: collect definitions and uses
    for (i, stmt) in enumerate(stmts)
        stmt isa Expr || continue
        assignment = _assignment(stmt)
        used_expr = stmt
        if !isnothing(assignment)
            (; lhs, rhs) = assignment
            if lhs isa Symbol && rhs isa Symbol
                alias_map[lhs] = rhs
            end
            used_expr = rhs
        end
        for symbol in walk_symbols(used_expr)
            last_use[symbol] = i
        end
    end

    for (alias, src) in alias_map
        alias_last_use = get(last_use, alias, 0)
        last_use[src] = max(get(last_use, src, 0), alias_last_use)
    end

    # `F_u = tmp1` aliases one NDArray under two names; resolve to a canonical rep
    # so it's freed once (double-free is masked only by the ptr=0 null-out).
    function canon(v)
        seen = Set{Symbol}()
        while haskey(alias_map, v) && !(v in seen)
            push!(seen, v)
            v = alias_map[v]
        end
        return v
    end

    function is_indexed_assign(stmt)
        assignment = _assignment(stmt)
        if isnothing(assignment)
            assignment = _broadcast_assignment(stmt)
        end
        if !isnothing(assignment)
            return !(assignment.lhs isa Symbol)
        end
        return false
    end
    function result_symbol(stmt)
        stmt isa Symbol && return stmt
        assignment = _assignment(stmt)
        isnothing(assignment) && return nothing
        assignment.lhs isa Symbol || return nothing
        return assignment.lhs
    end

    # Pass 2: insert finalizers
    out = Any[]
    n = length(stmts)
    freed = Set{Symbol}()

    # The block's value escapes to the caller, except for `A[...] = rhs`: Julia
    # returns `rhs` there, but that's a dead temp nobody consumes — free it and
    # return `nothing` rather than leak it or hand back a dangling handle.
    terminal_indexed = n > 0 && is_indexed_assign(stmts[n])

    protected = nothing
    if n > 0 && !terminal_indexed
        rs = result_symbol(stmts[n])
        if rs isa Symbol
            protected = canon(rs)
        end
    end

    function emit_delete!(v)
        c = canon(v)
        if c in freed || c == protected
            return nothing
        end
        push!(freed, c)
        push!(out, :(cuNumeric.maybe_insert_delete($v)))
        return nothing
    end

    for (i, stmt) in enumerate(stmts)
        # `v = w` aliases w, so don't finalize w at this statement.
        aliased_source = nothing
        assignment = _assignment(stmt)
        if !isnothing(assignment)
            (; lhs, rhs) = assignment
            if lhs isa Symbol && rhs isa Symbol
                aliased_source = rhs
            end
        end

        if i == n && !terminal_indexed
            res_var = Symbol(:res, counter[])
            counter[] += 1
            push!(out, :($res_var = $stmt))
        else
            push!(out, stmt)
        end

        for (v, lasti) in last_use
            if lasti == i && v in assigned_vars && v != aliased_source
                emit_delete!(v)
            end
        end

        if i == n
            push!(out, terminal_indexed ? :nothing : res_var)
        end
    end

    return out
end

"""
    insert_finalizers(block::Expr)
Apply finalizer insertion to a `begin ... end` or `:block` expression.
"""
function insert_finalizers(block::Expr, assigned_vars::Set{Symbol})
    stmts = _scope_statements(block)
    isnothing(stmts) && error("Expected a begin/block expression")
    return Expr(:block, insert_finalizers(stmts, assigned_vars)...)
end

function _process_lifetime_scope(scope, rewrite_lifetimes)
    try
        rewritten, assigned_vars = rewrite_lifetimes(scope)
        return insert_finalizers(rewritten, assigned_vars)
    finally
        counter[] = 0
    end
end

# Package-specific passes. The broadcast-aware pass also consumes the generic
# inter-broadcast fusion module included above.
include("lifetimes.jl")
include("broadcast_lifetimes.jl")

function process_ndarray_scope(scope; on_rewrite=nothing)
    # Broadcast expressions stay lazy only when fusion is enabled; otherwise
    # every call is analyzed as an eager allocation.
    @static if FUSE_BROADCAST_EXPRS
        return process_broadcast_lifetime_scope(scope; on_rewrite)
    end
    return process_lifetime_scope(scope)
end

# Return the deleted value for a generated finalizer call.
function _delete_argument(expr)
    call = _call(expr)
    isnothing(call) && return nothing
    if call.f != :(cuNumeric.maybe_insert_delete)
        return nothing
    end
    return only(call.args)
end

function print_lifetime_analysis(block; io::IO=stdout)
    rule = "-"^60
    stmts = _flatten_statements(process_ndarray_scope(block))
    mode = FUSE_BROADCAST_EXPRS ? "fusion-aware" : "plain"

    println(io, "@analyze_lifetimes expansion ($mode analysis)\n", rule)

    n = 0
    for s in stmts
        deleted = _delete_argument(s)
        if !isnothing(deleted)
            printstyled(io, lpad("✗ free ", 11), deleted, "\n"; color=:red)
        else
            n += 1
            println(io, lpad(n, 4), "  ", s)
        end
    end

    println(io, rule)
    return nothing
end

@doc"""
    @show_lifetimes expr

Print the lifetime-analysis rewrite of `expr` — the same transformation
[`@analyze_lifetimes`](@ref) applies — without running it. Every statement is
shown in source order and each inserted `maybe_insert_delete` is highlighted so
you can see exactly where each temporary is freed. Pure AST work, so it runs on
CPU-only checkouts.
"""
macro show_lifetimes(block)
    return :(print_lifetime_analysis($(QuoteNode(block))))
end
