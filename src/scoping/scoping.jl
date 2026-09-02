export @accelerate, @show_lifetimes

# Include generic syntax layers before the cuNumeric-specific lifetime passes.
include("util.jl")
using .ScopingUtils

include("inter_broadcast_fusion.jl")

const _DOT_MACRO_NAME = Symbol("@__dot__")

_is_dot_macro(::Any) = false
_is_dot_macro(name::Symbol) = name === _DOT_MACRO_NAME
_is_dot_macro(name::GlobalRef) = name.name === _DOT_MACRO_NAME

function _is_dot_macro(name::Expr)
    name.head === :. || return false
    quoted_name = last(name.args)
    return quoted_name isa QuoteNode && quoted_name.value === _DOT_MACRO_NAME
end

_expand_dot_macros(value, ::Module) = value

function _expand_dot_macros(expr::Expr, caller::Module)
    expr.head in (:quote, :inert) && return expr

    if expr.head === :macrocall && _is_dot_macro(first(expr.args))
        expanded = Base.macroexpand(caller, expr; recursive=false)
        return _expand_dot_macros(expanded, caller)
    end

    args = map(arg -> _expand_dot_macros(arg, caller), expr.args)
    return Expr(expr.head, args...)
end

# A block yields its final expression, not a `return` (which exits the enclosing
# function). Accept `return expr` only as the final statement and rewrite it to a
# plain trailing expression so it flows through analysis like any other value.
function _normalize_return(block)
    if block isa Expr && block.head === :return
        return isempty(block.args) ? :nothing : only(block.args)
    end
    stmts = _scope_statements(block)
    isnothing(stmts) && return block
    for (i, stmt) in enumerate(stmts)
        stmt isa Expr && stmt.head === :return || continue
        i == length(stmts) || throw(
            ArgumentError("`return` is only allowed as the final statement")
        )
        value = isempty(stmt.args) ? :nothing : only(stmt.args)
        return Expr(block.head, stmts[1:(end - 1)]..., value)
    end
    return block
end

const counter = Ref(0)

function maybe_insert_delete(var::NDArray)
    return cuNumeric.destroy!(var)
end

maybe_insert_delete(x) = x

# Symbols bound by an assignment anywhere in `expr` (let-form locals; soft-form
# protected bindings).
function _assigned_symbols(expr)
    assigned = Set{Symbol}()

    function collect_binding(lhs)
        if lhs isa Symbol
            push!(assigned, lhs)
        elseif lhs isa Expr && lhs.head in (:tuple, :parameters)
            foreach(collect_binding, lhs.args)
        elseif lhs isa Expr && lhs.head in (:(::), :(...))
            collect_binding(first(lhs.args))
        end
        return nothing
    end

    function visit(node)
        node isa Expr || return nothing
        assignment = _assignment(node)
        isnothing(assignment) || collect_binding(assignment.lhs)
        foreach(visit, node.args)
        return nothing
    end

    visit(expr)
    return assigned
end

function _lexical_scope(body, bindings::Set{Symbol})
    ordered = Base.sort!(collect(bindings); by=string)
    return Expr(:let, Expr(:block, ordered...), body)
end

function _hoist_temporary(expr, assigned_vars)
    counter[] += 1
    temporary = Symbol(:tmp, counter[])
    push!(assigned_vars, temporary)
    return temporary, [:($temporary = $expr)]
end

# Symbols a statement yields by reference (bare name or tuple elements): the
# values that escape the block, so they are exempt from the free and never fused.
function _result_symbols(stmt)
    stmt isa Symbol && return Set([stmt])

    assignment = _assignment(stmt)
    if !isnothing(assignment)
        return _result_symbols(assignment.rhs)
    end

    if stmt isa Expr && stmt.head in (:tuple, :parameters)
        return mapreduce(_result_symbols, union, stmt.args; init=Set{Symbol}())
    end
    if stmt isa Expr && stmt.head == :kw
        return _result_symbols(last(stmt.args))
    end
    if stmt isa Expr && stmt.head == :(::)
        return _result_symbols(first(stmt.args))
    end

    return Set{Symbol}()
end

function _returned_symbols(scope)
    stmts = _scope_statements(scope)
    (isnothing(stmts) || isempty(stmts)) && return Set{Symbol}()
    return _result_symbols(last(stmts))
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
function insert_finalizers(
    exprs::Vector, assigned_vars::Set{Symbol}; protected_roots::Set{Symbol}=Set{Symbol}()
)
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
    # Pass 2: insert finalizers
    out = Any[]
    n = length(stmts)
    freed = Set{Symbol}()

    # The block's value escapes to the caller, except for `A[...] = rhs`: Julia
    # returns `rhs` there, but that's a dead temp nobody consumes — free it and
    # return `nothing` rather than leak it or hand back a dangling handle.
    terminal_indexed = n > 0 && is_indexed_assign(stmts[n])

    # Roots (function args) are protected regardless of the terminal statement.
    protected = Set{Symbol}(canon(root) for root in protected_roots)
    if n > 0 && !terminal_indexed
        for result in _result_symbols(stmts[n])
            push!(protected, canon(result))
        end
    end

    function emit_delete!(v)
        c = canon(v)
        if c in freed || c in protected
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
function insert_finalizers(
    block::Expr, assigned_vars::Set{Symbol}; protected_roots::Set{Symbol}=Set{Symbol}()
)
    stmts = _scope_statements(block)
    isnothing(stmts) && error("Expected a begin/block expression")
    return Expr(:block, insert_finalizers(stmts, assigned_vars; protected_roots)...)
end

function _process_lifetime_scope(
    scope, rewrite_lifetimes; protected_roots::Set{Symbol}=Set{Symbol}()
)
    try
        rewritten, assigned_vars = rewrite_lifetimes(scope)
        return insert_finalizers(rewritten, assigned_vars; protected_roots)
    finally
        counter[] = 0
    end
end

# Package-specific passes. The broadcast-aware pass also consumes the generic
# inter-broadcast fusion module included above.
include("lifetimes.jl")
include("broadcast_lifetimes.jl")

function process_ndarray_scope(
    scope; on_rewrite=nothing, protected_roots::Set{Symbol}=Set{Symbol}()
)
    # Broadcast expressions stay lazy only when fusion is enabled; otherwise
    # every call is analyzed as an eager allocation.
    @static if FUSE_BROADCAST_EXPRS
        return process_broadcast_lifetime_scope(scope; on_rewrite, protected_roots)
    end
    return process_lifetime_scope(scope; protected_roots)
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

# Header + body statements per form, so the printout mirrors the real expansion.
function _analysis_parts(ex)
    ex = _strip_lines(ex)
    if ex isa Expr && ex.head === :function
        return "function " * string(first(ex.args)), _flatten_statements(ex.args[2])
    elseif ex isa Expr && ex.head === :let
        binds = _strip_lines(first(ex.args))
        bindstr = binds isa Expr ? join(binds.args, ", ") : string(binds)
        return "let " * bindstr, _flatten_statements(ex.args[2])
    end
    return nothing, _flatten_statements(ex)
end

# Pretty-print `expansion` (the exact `@accelerate` output), highlighting frees.
function print_lifetime_analysis(expansion; io::IO=stdout)
    rule = "-"^60
    header, stmts = _analysis_parts(expansion)
    mode = FUSE_BROADCAST_EXPRS ? "fusion-aware" : "plain"

    println(io, "@accelerate expansion ($mode)\n", rule)
    isnothing(header) || println(io, header)

    n = 0
    for s in stmts
        deleted = _delete_argument(s)
        if !isnothing(deleted)
            printstyled(io, lpad("✗ free ", 11), deleted, "\n"; color=:red)
        else
            n += 1
            println(io, lpad(n, 4), "  ", _strip_lines(s))
        end
    end

    isnothing(header) || println(io, "end")
    println(io, rule)
    return nothing
end

include("accelerate.jl")
